from survey import AnonymousSurvey
import pytest

# Define a question, and make a survey.
question = "What language did you first learn to speak?"
my_survey = AnonymousSurvey(question)

# # Show the question, and store responses to the question.
# my_survey.show_question()
# print("Enter 'q' at any time to quit.\n")
# while True:
#     response = input("Language: ")
#     if response == 'q':
#         break
#     my_survey.store_response(response)
#
# # Show the survey results.
# print("\nThank you to everyone who participated in the survey!")
# my_survey.show_results()

def test_store_single_response():
    """Test the store_response method."""
    question = "What is your favorite programming language?"
    survey = AnonymousSurvey(question)
    responses = ["Python", "JavaScript", "C++"]
    for response in responses:
        survey.store_response(response)

    for response in responses:
        assert response in survey.responses



@pytest.fixture
def language_survey():
    """Fixture to create a survey instance."""
    question = "What language did you first learn to speak?"
    language_survey = AnonymousSurvey(question)
    return language_survey

def test_store_multiple_responses(language_survey):
    """Test storing multiple responses."""
    responses = ["English", "Spanish", "French"]
    for response in responses:
        language_survey.store_response(response)

    assert len(language_survey.responses) == len(responses)
    for response in responses:
        assert response in language_survey.responses

def test_store_single_response(language_survey):
    """Test storing a single response."""
    response = "English"
    language_survey.store_response(response)
    assert len(language_survey.responses) == 1
    assert language_survey.responses[0] == response