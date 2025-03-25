import json
import os
from openai import OpenAI

class BatchProcessor():
    def __init__(self):
        self.client = OpenAI()

    def send_batch_file(self, filename, description):
        """
        Sends JSONL batch file to openai API for processing

        Parameters:
        - filename (str): The name of the output file.
        - description (str): Description of this batch.

        Returns:
        - bool: True if the file was created successfully, False otherwise.
        """

        batch_input_file = self.client.files.create(
        file=open(f"./Data/BatchFiles/{filename}", "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        meta_data = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": description
            }
        )

        print(f'Created batch with id: {meta_data.id}')
        
        return meta_data


    def fetch_batch(self, batch_id, output_file_name):
        try:
            # Retrieve the batch details
            batch_response = self.client.batches.retrieve(batch_id)
            output_file_id = batch_response.output_file_id

            if not output_file_id:
                raise ValueError(f"No output file associated with batch {batch_id}")

            # Retrieve the output file content
            file_response = self.client.files.content(output_file_id)
            file_content = file_response.read()  # Read the binary content from the response

            # Save the file content to a specified output file
            output_file_path = f"./Data/ResponseFiles/{output_file_name}"
            with open(output_file_path, 'wb') as f:
                f.write(file_content)  # Write the binary data to the file

            print(f"Batch {batch_id} has been successfully saved to {output_file_path}.")
        except Exception as e:
            print(f"An error occurred while fetching batch {batch_id}: {e}")

    def check_batch_status(self, batch_id):
        print(self.client.batches.retrieve(batch_id))

    def check_all_batch_status(self, limit=10):
        print(self.client.batches.list(limit=limit))

    '''Turns a response file into a dictionary with logprobs.'''
    def response_to_dict(self, response_file):
        # Initialize the nested dictionary structure
        grouped_responses = {}

        # Assuming each line in the JSONL file is a JSON object
        with open(f'./Data/ResponseFiles/{response_file}', 'r') as f:
            for line in f:
                # Parse each line as a JSON object
                data = json.loads(line)
                
                # Assuming custom_id has the structure {YEAR - STATE - PARTY - ITERATION}
                custom_id = data.get('custom_id')  # Retrieve the custom_id from the JSON data
                response = data.get('response')  # Retrieve the response from the JSON data
                
                if custom_id:
                    # Split and strip the custom_id into its components
                    keys = list(map(str.strip, custom_id.split('-')))
                    
                    # Start from the top level of grouped_responses
                    current_level = grouped_responses
                    
                    # Iterate through the keys to build the nested dictionary
                    for key in keys:
                        if key not in current_level:
                            current_level[key] = {}
                        current_level = current_level[key]
                    
                    # Now, current_level refers to the innermost dictionary based on the keys
                    logprobs = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs']
                    
                    for logprob in logprobs:
                        token = logprob.get('token')  # Safely get the 'token' key
                        prob = logprob.get('logprob') # Safely get the 'logprob' key
                        
                        if token and token.isdigit():
                            token = int(token)
                            current_level[token] = prob    
                        else:
                            current_level[token] = prob  

        return grouped_responses

    def response_to_df(self, response_file, category_labels=None):
        """
        Converts a JSONL response file to a pandas DataFrame with dynamic category handling.

        Parameters:
        - response_file (str): The filename of the JSONL response file located in './Data/ResponseFiles/'.
        - category_labels (list of str, optional): Custom labels for the categories extracted from 'custom_id'.
        If not provided, default labels like 'Category1', 'Category2', etc., will be used.

        Returns:
        - pd.DataFrame: A DataFrame containing the extracted data.
        """
        import pandas as pd
        
        # List to hold rows for the DataFrame
        data_list = []

        # Path to the response file
        file_path = f'./Data/ResponseFiles/{response_file}'

        # Open and read the JSONL file
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    # Parse each line as a JSON object
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"JSON decoding failed at line {line_number}: {e}")
                    continue  # Skip malformed JSON lines

                # Retrieve the 'custom_id' from the JSON data
                custom_id = data.get('custom_id')
                if not custom_id:
                    print(f"'custom_id' missing at line {line_number}. Skipping.")
                    continue  # Skip entries without 'custom_id'

                # Split the 'custom_id' into components and strip whitespace
                keys = [key.strip() for key in custom_id.split('-')]

                # Determine column labels for the categories
                if category_labels:
                    if len(category_labels) < len(keys):
                        # Extend the provided labels if there are more keys
                        labels = category_labels + [f'Category{i}' for i in range(len(category_labels)+1, len(keys)+1)]
                    else:
                        # Truncate the labels if there are fewer keys
                        labels = category_labels[:len(keys)]
                else:
                    # Use default labels if none are provided
                    labels = [f'Category{i}' for i in range(1, len(keys)+1)]

                # Extract the 'response' content
                try:
                    response_content = data['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Failed to extract 'response' at line {line_number}: {e}")
                    response_content = None  # Assign None if extraction fails

                # Extract the 'logprobs'
                try:
                    top_logprobs = data['response']['body']['choices'][0]['logprobs']['content'][0]['top_logprobs']
                    # Convert list of logprobs to a single dictionary
                    logprob_dict = {}
                    for lp in top_logprobs:
                        token = lp.get('token')
                        prob = lp.get('logprob')
                        if token is not None and prob is not None:
                            logprob_dict[token] = prob
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Failed to extract 'logprobs' at line {line_number}: {e}")
                    logprob_dict = {}  # Assign empty dict if extraction fails

                # Create a row dictionary with category data
                row = {label: key for label, key in zip(labels, keys)}
                row['Response'] = response_content
                row['Logprobs'] = logprob_dict

                # Optionally, convert 'Response' to numeric if it's a percentage
                # row['Response'] = pd.to_numeric(row['Response'], errors='coerce')

                data_list.append(row)

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data_list)

        return df

    def get_model(self, response_file):
        """
        Gets the model used

        Parameters:
        - response_file (str): Name of the jsonl file to the JSONL file.

        Returns:
        - dict: The first JSON object in the file.
        """
        with open(f'./Data/ResponseFiles/{response_file}', 'r') as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError("The JSONL file is empty.")
            try:
                data = json.loads(first_line)

                model = data['response']['body']['model']
                
                return model
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in the first line: {e}")
    
    def filter_candidates_jsonl(self, candidate1, candidate2, new_filename, current_filename):
        """
        Filters a JSONL file to include only entries for the specified candidates.

        Parameters:
        - candidate1 (str): The first candidate's name.
        - candidate2 (str): The second candidate's name.
        - new_filename (str): The name of the output JSONL file.
        - current_filename (str): The name of the input JSONL file.

        Raises:
        - ValueError: If either candidate is not found in the input file.
        - FileNotFoundError: If the input file does not exist.
        - json.JSONDecodeError: If a line in the input file is not valid JSON.
        """

        # Initialize a set to track found candidates
        found_candidates = set()

        # List to store filtered JSON objects
        filtered_entries = []

        try:
            with open(current_filename, 'r', encoding='utf-8') as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(f"Invalid JSON on line {line_number}: {e.msg}", e.doc, e.pos)

                    custom_id = data.get("custom_id", "")
                    if not custom_id:
                        continue  # Skip if custom_id is missing

                    # Split the custom_id to extract the candidate name
                    parts = custom_id.split(" - ")
                    if len(parts) < 4:
                        continue  # Skip if custom_id does not have enough parts

                    candidate_name = parts[2].strip()

                    if candidate_name == candidate1 or candidate_name == candidate2:
                        filtered_entries.append(data)
                        found_candidates.add(candidate_name)

            # Check if both candidates were found
            missing_candidates = {candidate1, candidate2} - found_candidates
            if missing_candidates:
                raise ValueError(f"Candidate(s) not found in the input file: {', '.join(missing_candidates)}")

            # Write the filtered entries to the new JSONL file
            with open(new_filename, 'w', encoding='utf-8') as outfile:
                for entry in filtered_entries:
                    json_line = json.dumps(entry)
                    outfile.write(json_line + '\n')

            print(f"Filtered JSONL file '{new_filename}' created successfully with candidates: {candidate1}, {candidate2}")

        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{current_filename}' does not exist.")
        except Exception as e:
            raise e

    def cancel_batch(self, batch_id):
        """
        Cancels a batch with the given batch_id.

        Parameters:
        - batch_id (str): The ID of the batch to cancel.

        Returns:
        - None
        """
        try:
            self.client.batches.cancel(batch_id)
            print(f"Batch {batch_id} has been successfully cancelled.")
        except Exception as e:
            print(f"An error occurred while cancelling batch {batch_id}: {e}")

if __name__ == "__main__":
    processor = BatchProcessor()
    batch_id_to_cancel = "batch_670d9f2e15cc819093c90b5e83cac1f5"
    processor.cancel_batch(batch_id_to_cancel)