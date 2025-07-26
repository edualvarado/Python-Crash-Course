from name_function import get_formatted_name

# print('Enter "q" at any time to quit.')
# while True:
#     first = input("\nPlease give me a first name: ")
#     if first == 'q':
#         break
#     last = input("Please give me a last name: ")
#     if last == 'q':
#         break
#
#     formatted_name = get_formatted_name(first, last)
#     print(f"\tNeatly formatted name: {formatted_name}")

def test_first_last():
    """Test the get_formatted_name function."""
    formatted_name = get_formatted_name('janis', 'joplin')
    assert formatted_name == 'Janis Joplin', f"Expected 'Janis Joplin', but got {formatted_name}"

    formatted_name = get_formatted_name('john', 'doe')
    assert formatted_name == 'John Doe', f"Expected 'John Doe', but got {formatted_name}"

    formatted_name = get_formatted_name('john', 'doe', 'michael')
    assert formatted_name == 'John Michael Doe', f"Expected 'John Michael Doe', but got {formatted_name}"

    print("All tests passed!")