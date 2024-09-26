
import os
import time
import requests


def timing_tracing_wrapper(func):
    def wrapper(*args, **kwargs):
        # simulate filesystem activity tracing
        separator = kwargs.get("separator")
        print(f'======= {separator}')
        os.path.isfile(f'/mnt/fs1/lroc/{separator}.txt')

        # run module and get execution time
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        running_time = end_time - start_time
        print(f"******** Running time for {func.__name__}: {running_time:.4f} seconds ********\n")

        return result

    return wrapper


@timing_tracing_wrapper
def read_url(url):
    response = requests.get(url)
    data = response.text

@timing_tracing_wrapper
def download_to_disk(url, file_path): 
    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else: 
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

@timing_tracing_wrapper
def read_from_disk(file_path):
    with open(file_path, 'rb') as f:
        f.read()

    

def main():

    url = 'https://snap.stanford.edu/data/amazon/Arts.txt.gz'
    # dest_folder = '/Users/leipan/Downloads/Datasets/AmazonReviews/'
    # filename = url.split('/')[-1].replace(" ", "_")  
    # file_path = os.path.join(dest_folder, filename)

    # download_to_disk(url, file_path, separator='download_to_disk')
    read_url(url, separator='read url to memory')
    
    # read_from_disk(file_path, separator='read from disk')

if __name__ == "__main__":
    main()
