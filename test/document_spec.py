from expects import expect, equal

from spamdetector.document import Document

with description('Document'):
    with it('should get the documet tokens as array'):
        document = Document(document_type='ham', document_data='Hello, how are you today?')

        example = document.get_document_example_tokens_array()

        expect(example).to(equal(['hello', 'how', 'are', 'you', 'today']))

    with it('should return the document type'):
        document = Document(document_type='ham', document_data='Hello, how are you today?')

        document_type = document.get_document_type()

        expect(document_type).to(equal('ham'))
