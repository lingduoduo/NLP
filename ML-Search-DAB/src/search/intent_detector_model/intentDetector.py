import time
import spacy
import Levenshtein as lev
import string
# from utils.logger.ml_std import LOGGER


class IntentDetector(object):
    def __init__(self, conf):
        """
        Initializes the IntentDetector object.

        This method loads the necessary data and models for intent detection.
        It sets up the prefix and ngram dictionaries for the Model 1 (Prefix Gram Model),
        and loads the English and multilingual models for the Model 2 (Spacy Models).

        Parameters:
            None

        Returns:
            None
        """
        # LOGGER.info("Model Initialization..")

        # loading the dataset
        action_data = conf["action_data"]
        position_data = conf["position_data"]
        people_data = conf["people_data"]
        common_action_typos = conf["common_action_typos"]

        # Setup for the Model 1 (Prefix Gram Model)
        stop_words = set(['of', 'the', 'and', 'with', 'at', 'to', 'from', 'on', ''])

        prefix_dict = dict()

        st = time.time()
        for i in range(len(action_data)):
            query = action_data[i].lower()  # this is single word

            if query not in stop_words:
                # Future Improvements: NGram Scorings
                # idx_list = ngram_dict.get(query, set())
                # idx_list.add(i)
                # ngram_dict[query] = idx_list

                # prefix gram
                for j in range(2, len(query) + 1):
                    qstr = query[:j]
                    idx_list = prefix_dict.get(qstr, set())
                    idx_list.add(i)
                    prefix_dict[qstr] = idx_list

        self.action_data = action_data
        self.position_data = position_data
        self.people_data = people_data
        self.prefix_dict = prefix_dict
        self.stop_words = stop_words
        self.common_action_typos = common_action_typos

        # Setup for the Model 2 (Prtefix Gram Model)
        english_model = spacy.load("en_core_web_sm")
        multilingual_model = spacy.load("xx_ent_wiki_sm")

        self.english_model = english_model
        self.multilingual_model = multilingual_model
        self.threshold = 5
        self.levenshtein_distance_threshold = 0.3

        # LOGGER.info(f"Time required to initialize the model: {time.time() - st} seconds")

    def calculate_levenshtein_distance(self, query):
        """
        Calculates the Levenshtein distance between each word in the query and the words that starts with same character in action data.
        Returns the top 3 minimum distances and the corresponding matched words with normalized distances.

        Args:
            query (str): The input query.

        Returns:
            dict: A dictionary containing the words from the query as keys and a list of tuples as values.
                  Each tuple contains the matched word, the Levenshtein distance, and the normalized distance.
        """
        result = {}
        for word in query.split():
            if len(word) > 3 and word not in self.stop_words:
                first_letter = word[0].lower()
                possible_words = [w for w in self.action_data if w.lower().startswith(first_letter)]

                distances = []
                for w in possible_words:
                    dist = lev.distance(word, w)
                    if dist == 0:
                        break
                    distances.append((w, dist, dist / len(word)))

                else:  # Only executed if the for loop didn't break
                    distances_less_than_threshold = []
                    if len(word) >= 6:
                        for d in distances:
                            if d[2] < self.levenshtein_distance_threshold:
                                distances_less_than_threshold.append(d)
                            else:
                                if word in self.common_action_typos:
                                    # add dist and normalized dist as 0
                                    distances_less_than_threshold = [(word, 0, 0)]
                    else: # executed for word length 4 or 5
                        if word in self.common_action_typos:
                            distances_less_than_threshold = [(word, 0, 0)]

                    top_3 = sorted(distances_less_than_threshold, key=lambda x: x[1])[:3]
                    if top_3:
                        result[word] = top_3
            else:
                if word in self.common_action_typos:
                    result[word] = [(word, 0, 0)]

        return result

    def query_score_prefixGram(self, query):
        """
        Calculates the score for the given query based on the prefix grams.

        Args:
            query (str): The input query string.

        Returns:
            int: The score calculated based on the prefix grams in the query.
        """
        stop_words = self.stop_words
        prefix_dict = self.prefix_dict

        # query = query.lower()
        words = query.lower().split(" ")
        words = [w for w in words if w not in stop_words]
        score_mapping = dict()
        for count, w in enumerate(words):
            word_score = 0
            # For N-Grams
            # factor_ngram = len(w)
            # if (w in ngram_dict):
            #     score += factor_ngram

            for j in range(2, len(w) + 1):
                qstr = w[:j]
                factor_prefixgram = len(qstr)

                if (qstr in prefix_dict):
                    # print(qstr, factor_prefixgram)
                    word_score += factor_prefixgram
            score_mapping[w] = word_score
        self.word_prefixGram_score = score_mapping
        return score_mapping

    def predict_category_prefixGram(self, query):
        """
        Predicts the category of a given query based on the prefix gram score.

        Parameters:
        query (str): The input query to be categorized.

        Returns:
        str: The predicted category ('people' or 'actions') based on the prefix gram score.
        """
        word_prefixGram_score = self.query_score_prefixGram(query)

        total_prefix_score_query = sum(word_prefixGram_score.values())

        levenshtein_distance_map = self.calculate_levenshtein_distance(query)
        self.levenshtein_distance_map = levenshtein_distance_map

        if total_prefix_score_query < self.threshold and len(levenshtein_distance_map) < 1:
            category_prefixGrams = 'people'
        else:
            category_prefixGrams = 'actions'

        return category_prefixGrams

    def predict_category_NER_POS_english(self, query):
        """
        Predicts the category of a given query using Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
        Using Spacy's English model -> en_core_web_sm
        (https://spacy.io/models/en#en_core_web_sm)

        Args:
            query (str): The input query to predict the category for.

        Returns:
            set: A set of categories predicted for the given query.
        """
        doc = self.english_model(str(query))
        category = set()

        noun_propn_entities = []
        for i, token in enumerate(doc):
            # most persons names would be identified as Proper Noun and sometimes as NOUN
            if (token.pos_ == "PROPN" or token.pos_ == "NOUN") and len(token.text) > 2 and self.word_prefixGram_score.get(token.text.lower(), 0) <= self.threshold:
                noun_propn_entities.append(token.text)

            # if query has intent for both action and query are usually prepositions followed by Person's name
            if token.tag_ in ["IN", "TO"] and i+1 < len(doc) and doc[i+1].pos_ in ["NOUN", "PROPN"] and not doc[i+1].text.lower() in self.action_data:
                category.add("people")

            if (token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "PROPN") and (token.text.lower() in self.action_data):
                category.add('actions')

            # For Positions
            if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and (token.text.lower() in self.position_data):
                category.add('people')

            # For most common people names
            if token.text.lower() in self.people_data:
                category.add('people')

            # For action words that might be missed due to typos (threshold of 27: 6 or more characters matched to action dataset)
            if self.word_prefixGram_score.get(token.text.lower(), 0) >= 27 and not (token.text.lower() in self.position_data) and not (token.text.lower() in self.people_data):
                category.add('actions')

            # if the word is in levenshtein_distance_map, then it is most likely a typo
            if len(self.levenshtein_distance_map) > 0:
                # delete that word from noun_propn_entities if it exists
                for word in noun_propn_entities:
                    if word in self.levenshtein_distance_map:
                        noun_propn_entities.remove(word)
                category.add('actions')

        for ent in doc.ents:
            if ent.label_ and ent.label_ != 'DATE':
                '''
                if NER is detected, then check if word is not in action data.
                NER detects first character captialized as one of the NER labels
                Example : "View" -> PERSON vs "view" -> NONE
                '''
                if ent.text.lower() not in self.action_data:
                    category.add('people')

        if len(noun_propn_entities) > 0:
            special_entities_category = self.predict_category_NER_entities(noun_propn_entities)
            # LOGGER.info(f"Category from detected Noun and ProperNoun Entities: {special_entities_category}")
            category.update(special_entities_category)

        # Call Multingual Model
        multilingual_category = self.predict_category_NER_multilingual(query)
        category.update(multilingual_category)

        return category

    def predict_category_NER_entities(self, noun_propn_entities):
        """
        Predicts the category of POS entities.

        Args:
            noun_propn_entities (list): A list of noun and proper noun entities.

        Returns:
            set: A set containing the predicted categories of the entities.
        """
        category = set()

        for word in noun_propn_entities:
            # capitalize the word and call the NER model
            word = word.capitalize()
            english_word = self.english_model(str(word))
            multilingual_word = self.multilingual_model(str(word))

            for ent in english_word.ents:
                if ent.label_ and ent.label_ != 'DATE':
                    if ent.text.lower() not in self.action_data:
                        category.add('people')
            for ent in multilingual_word.ents:
                if ent.label_ and ent.label_ != 'DATE':
                    if ent.text.lower() not in self.action_data:
                        category.add('people')
        return category

    def predict_category_NER_multilingual(self, query):
        """
        Predicts the category of a given query using Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
        Using Spacy's Multilingual model -> xx_ent_wiki_sm
        (https://spacy.io/models/xx#xx_ent_wiki_sm)

        Args:
            query (str): The input query to predict the category for.

        Returns:
            set: A set of categories predicted for the given query.
        """
        # check query if any word in the query is less than 3 characters
        # Multilingual fails to detect NER correctly
        words = query.split()
        words = [word for word in words if len(word) >= 3 and (self.word_prefixGram_score.get(word.lower(), 0) < self.threshold) and (word not in self.stop_words) and (word not in self.levenshtein_distance_map)]
        query = ' '.join(words)

        doc = self.multilingual_model(query)
        multilingual_category = set()

        for ent in doc.ents:
            if ent.label_ and ent.label_ != 'DATE':
                '''
                if NER is detected, then check if word is not in action data.
                NER detects first character captialized as one of the NER labels
                Example : "View" -> PERSON vs "view" -> NONE
                '''
                if ent.text.lower() not in self.action_data:
                    multilingual_category.add('people')

        return multilingual_category

    def preprocess_query(self, query):
        """
        Preprocesses the input query by removing any special characters and extra spaces.

        Args:
            query (str): The input query to preprocess.

        Returns:
            str: The preprocessed query string.
        """
        # trim down the leading and trailing spaces from the query
        query = query.strip()
        # Remove punctuation from the query
        query = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # removing single character 'a' and extra spaces if any present in the query words
        query = ' '.join([word for word in query.split() if word != 'a'])
        # encode the string
        query = query.encode('unicode_escape').decode()
        return query

    def predictModel(self, query: str):
        """
        Predicts the intent of a given query.

        Args:
            query (str): The input query to predict the intent for.

        Returns:
            dict: A dictionary containing the predicted intent categories.
                The keys represent different categories, and the values indicate
                whether the query belongs to that category or not. The possible
                categories are "actions", "people", and "teams".

        Examples:
            >>> detector = IntentDetector()
            >>> query = "view Bob's profile"
            >>> result = detector.predictModel(query)
            >>> print(result)
            {'actions': True, 'people': True, 'teams': False}
        """

        # LOGGER.info(f"Intent Detection for Query: {query}")

        response = {
            "actions": False,
            "people": False,
            "teams": False
        }

        # preprocessing the input query
        query = self.preprocess_query(query)

        if "team" in query.lower():
            response['teams'] = True

        # Quick Lookup for Category
        words = query.lower().split()
        # Check if every word in the query is in action dataset
        if all(word in self.action_data for word in words):
            response['actions'] = True
            # LOGGER.info("Prediction returned from Action Quick Lookup")
            return response

        if all(word in self.position_data for word in words):
            response['people'] = True
            # LOGGER.info("Prediction returned from Position Quick Lookup")
            return response

        # return all categories as True if the query is single word and has less than equal to 2 characters
        if len(query.split(" ")) == 1 and len(query) <= 2:
            response['actions'] = True
            response['people'] = True
            response['teams'] = True
            # LOGGER.info("Prediction returned for query less than 2 characters")
            return response

        category_prefixGrams = self.predict_category_prefixGram(query)
        # LOGGER.info(f"Category Prediction from PrefixGram Model : {category_prefixGrams}")

        if category_prefixGrams == "people":
            response["people"] = True
        else:
            # call the NER and POS Model
            category_NER_POS = self.predict_category_NER_POS_english(query)
            # LOGGER.info(f"Category Prediction from Spacy Models : {category_NER_POS}")

            for cat in category_NER_POS:
                response[cat] = True

            # check if response cat is all false to account for typos in action words
            if all(value == False for value in response.values()):
                if len(words) < 2:
                    word_score = self.word_prefixGram_score.get(words[0], 0)
                    if word_score >= self.threshold:
                        response["actions"] = True
                else:
                    # check the words within query to be in action_data, most likely the cat would be actions
                    for word in words:
                        if word in self.action_data:
                            response["actions"] = True
                            break
                        else:
                            response["people"] = True

        return response