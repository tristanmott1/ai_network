from BatchProcessor import BatchProcessor
from BatchGenerator import BatchGenerator

import argparse
import os

def genbatch(gen_method, filename, max_tokens=100, model='gpt-4o-mini'):
    batch_generator = BatchGenerator()
    if hasattr(batch_generator, gen_method):
        method_to_call = getattr(batch_generator, gen_method)
        batch_messages, labels = method_to_call()
        if batch_generator.create_json_batch_file(filename=f"{filename}", batch_messages=batch_messages, labels=labels, max_tokens=max_tokens, model=model):
            print(f"Created batchfile at ./BatchFiles/{filename}")
    else:
        print(f"Method '{gen_method}' not found.")
    

def sendbatch(file_name, description):
    batch_processor = BatchProcessor()
    if os.path.exists(f'./Data/BatchFiles/{file_name}'):
        meta_data = batch_processor.send_batch_file(file_name, description)
    else:
        print(f'./Data/BatchFiles/{file_name} not found')

def checkbatch(batch_id="all"):
    batch_processor = BatchProcessor()
    if batch_id == "all":
        batch_processor.check_all_batch_status()
    else:
        batch_processor.check_batch_status(batch_id)

def getbatch(batch_id, file_name):
    """
    Fetches a batch by ID and saves it to a file using the BatchProcessor class.

    Parameters:
    - batch_processor (BatchProcessor): An instance of the BatchProcessor class.
    - batch_id (str): The ID of the batch to fetch.
    - file_name (str): The name of the file to save the batch content.

    Returns:
    - None: The result is saved to a file.
    """
    batch_processor = BatchProcessor()
    batch_processor.fetch_batch(batch_id, file_name)




def main():
    parser = argparse.ArgumentParser(description="Batch Processing Program")
    subparsers = parser.add_subparsers(dest="command")

    # genbatch command
    genbatch_parser = subparsers.add_parser('genbatch', help="Generate a batch")
    genbatch_parser.add_argument('gen_method', type=str, help="Name of Batch Generator Function")
    genbatch_parser.add_argument('file_name', type=str, help="The name of the batch file to generate. Do not include file extension.")
    genbatch_parser.add_argument('--model', type=str, default='gpt-4o-mini', help="Model to use (default gpt-4o-mini)")
    genbatch_parser.add_argument('--max_tokens', type=int, default=100, help="Max output tokens (default is 100)")

    # sendbatch command
    sendbatch_parser = subparsers.add_parser('sendbatch', help="Send a batch file")
    sendbatch_parser.add_argument('file_name', type=str, help="The name of the batch file to send")
    sendbatch_parser.add_argument('--desc', type=str, default="LLM voting batch file", help="Description of batch file")

    # checkbatch command
    checkbatch_parser = subparsers.add_parser('checkbatch', help="check the status of a batch file")
    checkbatch_parser.add_argument('--batch_id', type=str, default="all", help="The ID of the batch to check. default checks all")

    # getbatch command
    getbatch_parser = subparsers.add_parser('getbatch', help="Get a batch file by ID")
    getbatch_parser.add_argument('batch_id', type=str, help="The ID of the batch to fetch")
    getbatch_parser.add_argument('file_name', type=str, help="Name to save batch response to")


    # target_party, jsonl, k=10000, iteration='0', verbose=False

    # Parse arguments
    args = parser.parse_args()

    # Dispatch to the appropriate function
    if args.command == "genbatch":
        genbatch(args.gen_method, args.file_name, args.max_tokens, args.model)
    elif args.command == "sendbatch":
        sendbatch(args.file_name, args.desc)
    elif args.command == "checkbatch":
        checkbatch(args.batch_id)
    elif args.command == "getbatch":
        getbatch(args.batch_id, args.file_name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()