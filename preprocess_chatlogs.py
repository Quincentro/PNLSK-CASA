import pandas as pd
import re
import pprint
from dataclasses import dataclass, field
from typing import List, Dict
pp = pprint.PrettyPrinter(indent=4)


@dataclass
class ProcessChatlogs:
    """
    This class transforms a list of transcripts into a list of sentences that can be used for nlp methods or transforms
    it into a dictionary that contains both the KLT/CSM and their sentences in the chat.
    """

    chatlogs: list
    chat_dict: Dict = field(default_factory=lambda: {})
    agent_error_logs: List = field(default_factory=lambda: [])
    client_error_logs: List = field(default_factory=lambda: [])

    def __post_init__(self):
        """"
        This function defines the regular expressions needed to extract information from the chatlogs
        """
        # This will be used to link the chatmessages to the corresponding sender KSM/HZ
        self.regex_agent_name = r'Agent\s(\w+)'
        self.regex_client_name = r'(.*?:\s)'

        # This will return all the timestamps
        # For example, (1m 30s) persoon: hoi allemaal (20m 20s) persoon hoe gaat het allemaal? -> [(1m 30s), (20m 20s)]
        self.regex_timestamps = r'\(\s\d{0,2}m?\(?d{0,2}s?\s?\d{1,2}s\s\)'

        # This will acquire all the text between two time slots ( 30s ) text ( 1m 30s)
        # This method doesn't acquire the last line of the text as there is no ending time slot.
        self.regex_without_last_response = r'\d{0,2}s?\s?\d{1,2}s\s\)\s{1}(.*?)\(\s\d{1}'

        # # This also requires the last response which will be captured in group 2
        self.regex_with_last_response = r'\d{0,2}s?\s?\d{1,2}s\s\)\s(.*?)\(\s|\s\)\s(.*$)'

    def apply_agent_regex(self, chatlog) -> str:
        """
        This function takes a chatlog as input and returns the persons in the conversation
        :returns: a tuple containing (CSM, KLT)
        """
        try:
            agent_name = re.findall(self.regex_agent_name, chatlog)[0]
            return agent_name
        # Some chatlogs don't have a response from the CSM, this wil result in an IndexError
        except IndexError:
            return 'No agent'

    def apply_client_regex(self, sentence) -> str:
        """
        This function returns the name of the client from a sentence
        :returns: a string, containing the name of the client
        """
        client_name = re.findall(self.regex_client_name, sentence)[0]
        return client_name

    def apply_timestamp_regex(self, chatlog) -> list:
        """
        # This function will return all the timestamps
        # For example, (1m 30s) persoon: hoi allemaal (20m 20s) persoon hoe gaat het allemaal? -> [(1m 30s), (20m 20s)]
        :returns: a list with all the timestamps of a specific chatlog.
        """
        # Acquire all the time stamps from the chatlog
        time_stamps = re.findall(self.regex_timestamps, chatlog)
        return time_stamps

    def apply_chat_sentence_regex(self, chatlog) -> list:
        """
        This function acquires all the text between timestamps
        """
        chat_sentences = re.findall(self.regex_with_last_response, chatlog)
        return chat_sentences

    def convert_timestamps_into_time(self) -> int:
        """
        The timestamps returned from the chatlogs are in string format and should be trandsformed to int/time in order
        to make calculations with timestamps
        :returns: an integer/time, displaying the time a sentence was typed in chat
        """
        pass

    def acquire_dialogue(self) -> dict:
        """
        This function combines the results of the regular expressions and creates a dialogue format that can be
        """
        for chat_index, chatlog in enumerate(self.chatlogs):
            sentences = []

            agent_name = self.apply_agent_regex(chatlog)

            # Skip the chatlog if there is no agent in the chatlog
            if agent_name == 'No agent':
                self.agent_error_logs.append(chatlog)
                continue

            # Acquire all the text from the chatlog via regex
            chat_sentences = self.apply_chat_sentence_regex(chatlog)

            # Acquire all the time stamps from the chatlog via regex
            time_stamps = self.apply_timestamp_regex(chatlog)

            # Get the sentence from the regex group that isn't empty denoted by {tex text text}
            # Group1 contains sentences between timestamps ( 30s ) {text text text} ( 1m 40s )
            # Group2 contains the last sentence in a chatlog whenever there is no second timestamp
            # ( 30s ) text text text ( 1m 40s ) {text text text}
            for sentence in chat_sentences:

                if sentence[0] == '':
                    sentences.append(sentence[1])

                if sentence[1] == '':
                    sentences.append(sentence[0])

            sentence_dict = {}

            # Define lists that store chatlines of both the CSM and the HZ
            agent_sentences = []
            client_sentences = []

            for sentence_index, (sentence, timestamp) in enumerate(zip(sentences, time_stamps)):

                # Capture the name of the KLT in a chatlog
                client_name = re.findall(self.regex_client_name, sentence)[0]

                # CSMs have two ways to identify themself in a conversation
                # For example, Persoon: Hallo ik ben persoon & Klantenservice: Hallo ik ben
                # This if statement ensures that CSM and HZ won't get mixed up
                if client_name == 'Klantenservice: ':
                    agent_name = 'Klantenservice'

                if f"{agent_name}: " in sentence:
                    agent_sentences.append((sentence_index, sentence.replace(f"{agent_name}: ", ''), timestamp))

                else:
                    client_sentences.append((sentence_index, sentence.replace(client_name, ''), timestamp))
                    actual_client_name = client_name

            # Create a dictionary for both the CSM/HZ to store the dialogue
            sentence_dict[agent_name] = agent_sentences
            sentence_dict[actual_client_name] = client_sentences

            # Add a conversation to chat_dict with chat_index as index
            # chat_index will be replaced with casenumber when we have actual data
            self.chat_dict[chat_index] = sentence_dict

        return self.chat_dict

    def find_chatlogs_without_client(self) -> list:
        """
        This function removes chatlogs where the agent is the only sender of messages in the chatlog
        :returns: a list, containing the indexes of the
        """
        chat_dict = self.acquire_dialogue()

        for key, dialogue in chat_dict.items():

            CSM, KLT = dialogue.keys()

            if len(dialogue[KLT]) == 0:
                self.client_error_logs.append(key)

        return self.client_error_logs

    def return_chatlogs(self) -> dict:
        """
        This function returns the chatlogs
        :returns: a dictionary, containing
        """
        # Acquire the indexes of the chatlogs where there is no client talking
        client_errors = self.find_chatlogs_without_client()
        # Remove the chatlogs where no client is talking based on the index
        final_chatlogs = {k: v for k, v in self.chat_dict.items() if k not in client_errors}

        return final_chatlogs


if __name__ == '__main__':

    df = pd.read_excel('voorbeelden_chatconversaties.xlsx', engine='openpyxl')

    # Define the list where all chatlogs have been stored
    # This has to be replaced when we have actual data.
    chatlogs = df['Body']

    PCL = ProcessChatlogs(chatlogs)

    chatlog_dialogues = PCL.return_chatlogs()
