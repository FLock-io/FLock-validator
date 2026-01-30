import requests
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def download_file(url: str) -> str:
    try:
        # Send a GET request to the signed URL
        response = requests.get(url)
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write the content to the temp file in binary mode
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)

            # move the file pointer to the beginning of the file
            temp_file.flush()
            temp_file.seek(0)

            # get the file path
            file_path = temp_file.name
            print(f"Downloaded the file to {file_path}")

            return file_path

    except requests.exceptions.RequestException as e:
        # Handle any exception that can be raised by the requests library
        print(f"An error occurred while downloading the file: {e}")
        raise e
