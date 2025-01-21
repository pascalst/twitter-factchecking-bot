import re
import tweepy
from airtable import Airtable
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import DuckDuckGoSearchRun
import schedule
import time
import os


# Helpful when testing locally
from dotenv import load_dotenv
load_dotenv()

# Load your Twitter and Airtable API keys (preferably from environment variables, config file, or within the railyway app)
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "YourKey")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "YourKey")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "YourKey")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "YourKey")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YourKey")

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY", "YourKey")
AIRTABLE_BASE_KEY = os.getenv("AIRTABLE_BASE_KEY", "YourKey")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "YourKey")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YourKey")

# TwitterBot class to help us organize our code and manage shared state
class TwitterBot:
    def __init__(self):
        self.twitter_api = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN,
                                         consumer_key=TWITTER_API_KEY,
                                         consumer_secret=TWITTER_API_SECRET,
                                         access_token=TWITTER_ACCESS_TOKEN,
                                         access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
                                         wait_on_rate_limit=True)

        self.airtable = Airtable(AIRTABLE_BASE_KEY, AIRTABLE_TABLE_NAME, AIRTABLE_API_KEY)
        self.twitter_me_id = self.get_me_id()
        self.tweet_response_limit = 35 # How many tweets to respond to each time the program wakes up

        # Initialize the language model w/ temperature of .3 to induce limited creativity and mostly fact-checking
        self.llm = ChatOpenAI(temperature=.3, openai_api_key=OPENAI_API_KEY, model_name='gpt-4')
        self.ddg_search = DuckDuckGoSearchRun()

        # For statics tracking for each run. This is not persisted anywhere, just logging
        self.mentions_found = 0
        self.mentions_replied = 0
        self.mentions_replied_errors = 0

    # Generate a response using the language model using the template we reviewed in the jupyter notebook (see README)
    def generate_response(self, mentioned_conversation_tweet_text):

        print (f"Generating response for: {mentioned_conversation_tweet_text}")

        system_template = """
            You are a highly meticulous fact-checker whose primary responsibility is to review and evaluate the accuracy of each post.

            % RESPONSE TONE:

            - Always polite, friendly, and respectful, fostering positive and engaging interactions.
            - Occasionally incorporate light humor, but never at the expense of clarity or accuracy.

            % RESPONSE FORMAT:

            - Provide concise, fact-focused responses.
            - Respond in under 200 characters
            - Do not respond with emojis
            - Back up statements with verifiable sources, if necessary.
            - Avoid speculation; redirect users to reliable resources (e.g., Wikipedia) if unsure.
            - Indicate your confidence level (High, Medium, Low) in the response.


            % RESPONSE CONTENT:

            - Prioritize accuracy and reliable information.
            - Stay on topic and approach arguments with precision and thorough analysis.
            - Praise users for correctly cited, lesser-known facts.
            - Respond in the language of the input. If unsure, default to English.
            - Maintain a politically neutral stance, emphasizing common sense and balance.
            - If the response requires further elaboration, suggest additional resources without overwhelming the user.
            - If you don't have an answer, say, "Sorry, the internet doesn't know this topic."
            - Include "Confidence: [High/Medium/Low]" at the end of your response always in English without a period following it.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template="{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # get a chat completion from the formatted messages
        final_prompt = chat_prompt.format_prompt(text=mentioned_conversation_tweet_text).to_messages()
        response = self.llm(final_prompt).content
        
        # Parse the confidence level from the response
        match = re.search(r"Confidence: (High|Medium|Low)\.?", response)
        confidence = match.group(1) if match else "Unknown"

        print("Confidence level of response:", confidence)
        
        if confidence == "Unknown":
            print("Warning: No confidence level detected in the response.")

        # If confidence is low, perform web search and generate a new response
        if confidence == "Low":
            print("Low confidence detected. Performing web search...")
            search_results = self.web_search(mentioned_conversation_tweet_text)
            print(f"Web search results: {search_results[:3]}")

            # Add search results to the new prompt
            search_context = f"Relevant web search results:\n{search_results[:3]}\n\n"
            updated_prompt = f"{search_context}{mentioned_conversation_tweet_text}"

            # Regenerate response with additional context
            final_prompt = chat_prompt.format_prompt(text=updated_prompt).to_messages()
            response = self.llm(final_prompt).content
        
        response = re.sub(r"Confidence: (High|Medium|Low)\.?", "", response).strip()

        print("Response generated:", response)

        return response

    def web_search(self, query):
        """Perform a DuckDuckGo search and return the top results."""
        print(f"Performing web search for: {query}")
        results = self.ddg_search.run(query)
        return results
    

        # Generate a response using the language model
    def respond_to_mention(self, mention, mentioned_conversation_tweet):
        response_text = self.generate_response(mentioned_conversation_tweet.text)
        
        # Try and create the response to the tweet. If it fails, log it and move on
        try:
            response_tweet = self.twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=mention.id)
            self.mentions_replied += 1
        except Exception as e:
            print (e)
            self.mentions_replied_errors += 1
            return
        
        # Log the response in airtable if it was successful
        print("Tweet response created:", response_tweet.data['id'])
        print("Logging response in airtable...")
        self.airtable.insert({
            'mentioned_conversation_tweet_id': str(mentioned_conversation_tweet.id),
            'mentioned_conversation_tweet_text': mentioned_conversation_tweet.text,
            'tweet_response_id': response_tweet.data['id'],
            'tweet_response_text': response_text,
            'tweet_response_created_at' : datetime.utcnow().isoformat(),
            'mentioned_at' : mention.created_at.isoformat()
        })
        return True
    
    # Returns the ID of the authenticated user for tweet creation purposes
    def get_me_id(self):
        return self.twitter_api.get_me()[0].id
    
    # Returns the parent tweet text of a mention if it exists. Otherwise returns None
    # We use this to since we want to respond to the parent tweet, not the mention itself
    def get_mention_conversation_tweet(self, mention):
        # Check to see if mention has a field 'conversation_id' and if it's not null
        if mention.conversation_id is not None:
            conversation_tweet = self.twitter_api.get_tweet(mention.conversation_id).data
            return conversation_tweet
        return None

    # Get mentioned to the user thats authenticated and running the bot.
    # Using a lookback window of 2 hours to avoid parsing over too many tweets
    def get_mentions(self):
        # If doing this in prod make sure to deal with pagination. There could be a lot of mentions!
        # Get current time in UTC
        now = datetime.utcnow()

        # Subtract 30 minutes to get the start time
        start_time = now - timedelta(minutes=30)

        # Convert to required string format
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print("Start logging into the twitterverse...")
        response = self.twitter_api.get_users_mentions(
            id=self.twitter_me_id,
            start_time=start_time_str,
            expansions=['referenced_tweets.id'],
            tweet_fields=['created_at', 'conversation_id']
        )

        # Extract rate limit info from the response
        rate_limit_remaining = response.meta.get('x-rate-limit-remaining', 'Unknown')
        rate_limit_reset = response.meta.get('x-rate-limit-reset', 'Unknown')
        
        print(f"Rate limit remaining: {rate_limit_remaining}")
        print(f"Rate limit reset time (Unix timestamp): {rate_limit_reset}")
        
        # Print raw mentions data
        print(f"Raw mentions data: {response.data}")
        
        return response.data

    # Checking to see if we've already responded to a mention with what's logged in airtable
    def check_already_responded(self, mentioned_conversation_tweet_id):
        records = self.airtable.get_all(view='Grid view')
        for record in records:
            if record['fields'].get('mentioned_conversation_tweet_id') == str(mentioned_conversation_tweet_id):
                return True
        return False

    # Run through all mentioned tweets and generate a response
    def respond_to_mentions(self):
        mentions = self.get_mentions()

        # If no mentions, just return
        if not mentions:
            print("No mentions found")
            return
        
        self.mentions_found = len(mentions)

        for mention in mentions[:self.tweet_response_limit]:
            # Getting the mention's conversation tweet
            mentioned_conversation_tweet = self.get_mention_conversation_tweet(mention)
            
            # If the mention *is* the conversation or you've already responded, skip it and don't respond
            if (mentioned_conversation_tweet.id != mention.id
                and not self.check_already_responded(mentioned_conversation_tweet.id)):

                self.respond_to_mention(mention, mentioned_conversation_tweet)
        return True
    
        # The main entry point for the bot with some logging
    def execute_replies(self):
        print (f"Starting Job: {datetime.utcnow().isoformat()}")
        self.respond_to_mentions()
        print (f"Finished Job: {datetime.utcnow().isoformat()}, Found: {self.mentions_found}, Replied: {self.mentions_replied}, Errors: {self.mentions_replied_errors}")

# The job that we'll schedule to run every X minutes
def job():
    print(f"Job executed at {datetime.utcnow().isoformat()}")
    bot = TwitterBot()
    bot.execute_replies()

def test_get_mentions():
    bot = TwitterBot()
    mentions = bot.get_mentions()
    if mentions:
        for mention in mentions:
            print(f"Mention ID: {mention.id}, Text: {mention.text}")
    else:
        print("No mentions found.")


# Method to test the web search
def test_web_search():
    bot = TwitterBot()
    query = "does elon musks mother live in china"
    results = bot.web_search(query)
    print("Web Search Results:")
    print(results)

# if __name__ == "__main__":
#     test_get_mentions()
#     test_web_search()



if __name__ == "__main__":
    # Schedule the job to run every 20 minutes. Edit to your liking, but watch out for rate limits
    schedule.every(20).minutes.do(job)
    print("Starting up all systems...")
    job()
    while True:
        schedule.run_pending()
        time.sleep(1)