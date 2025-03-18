def clean_yaml_file(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()

    # Remove null characters
    cleaned_content = content.replace(b'\x00', b'')

    with open(file_path, 'wb') as file:
        file.write(cleaned_content)

# Path to your environment.yml file
file_path = 'C:/Users/vicvi/BeliefNetworkEvo/pythons-beliefs.yml'

clean_yaml_file(file_path)