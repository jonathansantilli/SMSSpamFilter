class InvalidDocumentTypeException(Exception):
    pass

class Document:
    SPAM = 'spam'
    HAM = 'ham'
    DOCUMENT_EXAMPLE_TYPES = [SPAM, HAM]
    TOKEN_SPLITTER_SYMBOLS = ['.', ',', ';', '(', ')', '-', '_', '!', '>', '<', '/', '+', ':', '?', '=', '@']

    def __init__(self, document_type:str, document_data:str):
        self.document_type = document_type
        self.document_data = document_data

        if not self._is_valid_document_example_type(document_type):
            raise InvalidDocumentTypeException("The document type '" + document_type + "' is invalid")

    def get_document_example_tokens_array(self) -> list:
        """
        Get the preprocessed documents as an Array

        :return Array: the document example
        """
        return [token for token in self._process_data().split(' ') if token]

    def get_document_type(self) -> str:
        """
        Get the document type

        :return str: the document type
        """
        return self.document_type

    def _is_valid_document_example_type(self, document_type_to_check) -> bool:
        """
        Validate if the document type is allowed among the two possible values

        :return bool: True in case is valid, False otherwise
        """

        return document_type_to_check in Document.DOCUMENT_EXAMPLE_TYPES

    def _process_data(self):
        """
        Process each document in order to unify and clean them up

        :return str: the cleaned document
        """
        processed_data = self.document_data.lower()
        for symbol in Document.TOKEN_SPLITTER_SYMBOLS:
            processed_data = ' '.join(processed_data.split(symbol)).strip()

        return processed_data
