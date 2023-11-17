#! /usr/bin/env python3
"""
Scripts for automatically downloading and filtering Lick Observatory skycamera images.
This occurs in two stages:
    1. Visit https://mthamilton.ucolick.org/data/ and download archived skycam images at a specified date
    2. Delete skycam images according to a filter rule
"""
import time
from datetime import datetime, timedelta, tzinfo
import os
import tarfile
import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By


from skycam_utils import get_skycam_dir, get_skycam_subdirs, is_data_downloaded, get_skycam_img_time


def get_skycam_link(skycam_type, year, month, day):
    """
    Skycam link formats
      0 = 4 digit year
      1 = 2 digit month
      2 = 2 digit day, 1-indexed
    """
    if skycam_type == 'SC':
        return f'https://mthamilton.ucolick.org/data/{year}-{month:0>2}/{day:0>2}/allsky/public/'
    elif skycam_type == 'SC2':
        return f'https://mthamilton.ucolick.org/data/{year}-{month:0>2}/{day:0>2}/skycam2/public/'



def download_wait(directory, timeout, nfiles=None, verbose=False):
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
                num_crdownload_files += 1
                current_fsize = os.path.getsize(f'{directory}/{fname}')
                if fname in last_download_fsizes:
                    last_fsize = last_download_fsizes[fname]
                    if current_fsize > last_fsize:
                        seconds = 0
                    last_download_fsizes[fname] = current_fsize
                else:
                    last_download_fsizes[fname] = current_fsize

        if num_crdownload_files == 0 and verbose:
            print('Successfully downloaded all files')

        dl_wait = num_crdownload_files > 0
        seconds += 1
    return seconds


def download_skycam_data(skycam_type, year, month, day, verbose, root):
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


    # Create skycam directory
    skycam_path = f'{root}/{get_skycam_dir(skycam_type, year, month, day)}'
    os.makedirs(skycam_path, exist_ok=True)

    # Set Chrome driver options
    prefs = {
        'download.default_directory': os.path.abspath(skycam_path)
    }
    #print(os.path.abspath(skycam_dir))
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('prefs', prefs)
    chrome_options.add_argument("--headless")   # Don't open a browser window
    # Open Chrome driver
    if verbose: print('Initializing Chrome webdriver...')
    driver = webdriver.Chrome(options=chrome_options)

    # Open URL to Lick skycam archive
    link = get_skycam_link(skycam_type, year, month, day)
    driver.get(link)

    # Check if download page exists
    title = driver.title
    if title == '404 Not Found' or title != 'Mt. Hamilton Data Repository':
        msg = f"The link '{link}' does not contain valid skycam data. Title of webpage at that link: '{title}'. Exiting..."
        driver.close()
        shutil.rmtree(skycam_path)
        raise Warning(msg)

    # Select all files to download
    select_all_button = driver.find_element(By.LINK_TEXT, "all")
    select_all_button.click()

    # Deselect movie files
    # These are always the first two checkboxes
    movie_checkboxes = driver.find_elements(By.XPATH, "//input[@type='checkbox']")
    for i in range(2):
        movie_checkboxes[i].click()

    # Download tarball file to out_dir
    print(f'Downloading data from {link}')
    download_tarball = driver.find_element(By.XPATH, "//input[@type='submit']")
    download_tarball.click()

    download_wait(directory=skycam_path, timeout=30, nfiles=1, verbose=verbose)
    driver.close()

    return skycam_path


def unzip_images(skycam_path):
    """Unpack image files from tarball."""
    img_subdirs = get_skycam_subdirs(skycam_path)
    downloaded_fname = ''

    for fname in os.listdir(skycam_path):
        if fname.endswith('.tar.gz'):
            downloaded_fname = fname
    if downloaded_fname:
        downloaded_fpath = f'{skycam_path}/{downloaded_fname}'
        with tarfile.open(downloaded_fpath, 'r') as tar_ref:
            tar_ref.extractall(skycam_path)
        os.remove(downloaded_fpath)

        for path in os.listdir(skycam_path):
            if os.path.isdir(f'{skycam_path}/{path}') and path.startswith('data'):
                os.rename(f'{skycam_path}/{path}', img_subdirs['original'])


def filter_images(skycam_dir: str, first_t: datetime, last_t: datetime):
    """Remove skycam images between t_start and t_end."""
    path_to_orig_skycam_imgs = get_skycam_subdirs(skycam_dir)['original']
    for fname in sorted(os.listdir(path_to_orig_skycam_imgs)):
        if fname.endswith('.mp4') or fname.endswith('.mpg'):
            os.remove("{0}/{1}".format(path_to_orig_skycam_imgs, fname))
        skycam_t = get_skycam_img_time(fname)
        if not (first_t <= skycam_t <= last_t):
            os.remove("{0}/{1}".format(path_to_orig_skycam_imgs, fname))


def unpack_and_filter_skycam_imgs(skycam_path, first_t, last_t, verbose=False):
    is_data_downloaded(skycam_path)
    if verbose: print("Unzipping skycam files...")
    unzip_images(skycam_path)
    if verbose: print("Filtering skycam images...")
    filter_images(skycam_path, first_t, last_t)


def get_manual_download_instructions(skycam_path, skycam_type, year, month, day):
    skycam_link = get_skycam_link(skycam_type, year, month, day)
    msg = (f'To manually download the data, please do the following:\n'
            f'\t0. Set manual_skycam_download=True in your call to the build_batch function\n'
            f'\t1. Visit {skycam_link}\n'
            f'\t2. Click the blue "all" text\n'
            f'\t3. Uncheck the .mp4 and .mpg files (should be near the top)\n'
            f'\t4. Click the "Download tarball" button and wait for the download to finish\n\n'
            f'\t(Please DO NOT unzip the file)\n\n'
            f'\t5. Move the downloaded .tar.gz file to \n'
            f'\t\t{os.path.abspath(skycam_path)}\n'
            f'\t6. Rerun make_batch.py')
    return msg

