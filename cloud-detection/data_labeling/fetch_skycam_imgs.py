#! /usr/bin/env python3
# Visit https://mthamilton.ucolick.org/data/ and download archived skycam
# images at a specified date
import time
from datetime import datetime, timedelta, tzinfo
import os
import tarfile

from selenium import webdriver
from selenium.webdriver.common.by import By


def get_skycam_link(skycam, year, month, day):
    """
    Skycam link formats
      0 = 4 digit year
      1 = 2 digit month
      2 = 2 digit day, 1-indexed
    """
    if skycam == 'SC':
        return f'https://mthamilton.ucolick.org/data/{year}-{month:0>2}/{day:0>2}/allsky/public/'
    elif skycam == 'SC2':
        return f'https://mthamilton.ucolick.org/data/{year}-{month:0>2}/{day:0>2}/skycam2/public/'


def get_out_dir(skycam, year, month, day):
    if skycam == 'SC':
        return f'SC_img_{year}-{month:0>2}-{day:}'
    elif skycam == 'SC2':
        return f'SC2_img_{year}-{month:0>2}-{day:}'


def download_wait(directory, timeout, nfiles=None):
    """
    Wait for downloads to finish with a specified timeout.

    Args
    ----
    directory : str
        The path to the folder where the files will be downloaded.
    timeout : int
        How many seconds to wait until timing out.
    nfiles : int, defaults to None
        If provided, also wait for the expected number of files.

    Code from: https://stackoverflow.com/questions/34338897/python-selenium-find-out-when-a-download-has-completed
    """
    seconds = 0
    dl_wait = True
    last_download_fsizes = {}
    while dl_wait and seconds < timeout:
        time.sleep(1)
        dl_wait = False
        files = os.listdir(directory)
        if nfiles and len(files) != nfiles:
            dl_wait = True

        num_crdownload_files = 0
        for fname in files:
            if fname.endswith('.crdownload'):
                #print(last_download_fsizes)
                num_crdownload_files += 1
                current_fsize = os.path.getsize(f'{directory}/{fname}')
                if fname in last_download_fsizes:
                    last_fsize = last_download_fsizes[fname]
                    if current_fsize > last_fsize:
                        seconds = 0
                    last_download_fsizes[fname] = current_fsize
                else:
                    last_download_fsizes[fname] = current_fsize

        if num_crdownload_files == 0:
            print('Successfully downloaded all files')

        dl_wait = num_crdownload_files > 0
        seconds += 1
    return seconds


def download_skycam_data(skycam, year, month, day):
    """
    Downloads all the skycam images collected on year-month-day from 12pm up to 12pm the next day (in PDT).

    @param skycam: 'SC' or 'SC2'
    @param year: 4 digit year,
    @param month: month, 1-indexed
    @param day: day, 1-indexed
    """
    assert len(str(year)) == 4, 'Year must be 4 digits'
    assert 1 <= month <= 12, 'Month must be between 1 and 12, inclusive'
    assert 1 <= day <= 31, 'Day must be between 1 and 31, inclusive'

    # Set Chrome driver options
    out_dir = get_out_dir(skycam, year, month, day)
    try:
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(f'{out_dir}/original'):
            raise FileExistsError(f"Data already downloaded at {out_dir}/original")
    except FileExistsError as e:
        print(e)
        return None
    prefs = {
        'download.default_directory': out_dir
    }
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('prefs', prefs)
    chrome_options.add_argument("--headless")   # Don't open a browser window
    # Open Chrome driver
    driver = webdriver.Chrome(options=chrome_options)

    # Open URL to Lick skycam archive
    link = get_skycam_link(skycam, year, month, day)
    driver.get(link)
    #print(f'link={link}, out_dir={out_dir}')

    # Check if download page exists
    title = driver.title
    if title == '404 Not Found' or title != 'Mt. Hamilton Data Repository':
        print(f"The link '{link}' does not contain valid skycam data. Exiting...")
        driver.close()
        return None

    # Select all files to download
    select_all_button = driver.find_element(By.LINK_TEXT, "all")
    select_all_button.click()

    # Deselect movie files
    # These are always the first two checkboxes
    movie_checkboxes = driver.find_elements(By.XPATH, "//input[@type='checkbox']")
    for i in range(2):
        movie_checkboxes[i].click()

    # Download tarball file to out_dir
    print('Downloading data...')
    download_tarball = driver.find_element(By.XPATH, "//input[@type='submit']")
    download_tarball.click()

    download_wait(directory=out_dir, timeout=30, nfiles=1)
    driver.close()

    return out_dir


def unzip_images(skycam_dir):
    """Unpack image files from tarball."""
    original_skycam_img_dir = f'{skycam_dir}/original'
    downloaded_fname = ''

    for fname in os.listdir(skycam_dir):
        if fname.endswith('.tar.gz'):
            downloaded_fname = fname
    if downloaded_fname:
        downloaded_fpath = f'{skycam_dir}/{downloaded_fname}'
        with tarfile.open(downloaded_fpath, 'r') as tar_ref:
            tar_ref.extractall(skycam_dir)
        os.remove(downloaded_fpath)

        for path in os.listdir(skycam_dir):
            if os.path.isdir(f'{skycam_dir}/{path}') and path.startswith('data'):
                os.rename(f'{skycam_dir}/{path}', original_skycam_img_dir)
    

def get_img_time(skycam_fname):
    """Returns datetime object based on the image timestamp contained in skycam_fname."""
    if skycam_fname[-4:] != '.jpg':
        raise Warning('Expected a .jpg file')
    # Example: SC2_20230625190102 -> SC2, 20230625190102
    skycam_type, t = skycam_fname[:-4].split('_')
    # 20230625190102 -> 2023, 06, 25, 19, 01, 02
    time_fields = t[0:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]
    year, month, day, hour, minute, second = [int(tf) for tf in time_fields]

    timestamp = datetime(year, month, day, hour, minute, second)
    return timestamp



def remove_day_images(skycam_dir):
    """Remove skycam images taken during the day."""
    original_skycam_img_dir = f'{skycam_dir}/original'
    PST_offset = timedelta(hours=-7)
    for skycam_fname in sorted(os.listdir(original_skycam_img_dir)):
        pst_time = get_img_time(skycam_fname) + PST_offset
        # Delete images taken between 5am and 8pm, inclusive
        if 5 <= pst_time.hour <= 12 + 8:
            #print(f'{original_skycam_img_dir}/{skycam_fname}')
            #print(os.path.exists(f'{original_skycam_img_dir}/{skycam_fname}'))
            os.remove(f'{original_skycam_img_dir}/{skycam_fname}')


def download_night_skycam_imgs(skycam, year, month, day):
    skycam_dir = download_skycam_data(skycam, year, month, day)
    if skycam_dir:
        unzip_images(skycam_dir)
        remove_day_images(skycam_dir)


download_night_skycam_imgs('SC2', 2023, 6, 25)
