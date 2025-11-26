from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import numpy as np
import os
import requests
import imageio_ffmpeg
import subprocess
from tinytag import TinyTag

from src.audio.signal import Signal

class NaxosScraper(object):

    UNKNOWN_DURATION = -1
    NA_DURATION = -2
    CONVERSION_FAILED = -3

    UNKNOWN_DURATION_TAG = "Unknown"
    NA_DURATION_TAG = "NA"

    def __init__(self, naxos_url, cache_dir="cache/naxos_data/"):
        # Naxos URL example: https://www.naxos.com/CatalogueDetail/?id=CHAN3019-21
        self.naxos_url = naxos_url
        self.id = naxos_url.split('=')[-1]
        self.cache_dir = os.path.join(cache_dir, self.id)
        self.audio_urls_file = os.path.join(self.cache_dir, "audio_URLs.npy")
        self.audio_full_durations_file = os.path.join(self.cache_dir, "audio_full_durations.npy")
        self.audio_titles_file = os.path.join(self.cache_dir, "audio_titles.npy")
        self.audio_files_path = os.path.join(self.cache_dir, "audio_files")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.audio_files_path, exist_ok=True)
    
    def mount_driver(self):
        # Setup Chrome with performance logging enabled
        options = Options()
        # options.add_argument("--headless=new") # comment this line to see the browser
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def _extract_content(self, button, pos, idx):
        try:
            # Find the parent <tr> element of the play button
            parent_tr = button.find_element(By.XPATH, "./ancestor::tr")
            # Find all <td> elements within this <tr>
            td_elements = parent_tr.find_elements(By.TAG_NAME, "td")
            # Get the last <td> which contains the duration
            if td_elements:
                duration = td_elements[pos].text.strip()
                print(f"Button {idx} duration: {duration}")
            else:
                duration = self.UNKNOWN_DURATION_TAG
                print(f"Button {idx}: No duration found")
        except Exception as e:
            duration = self.NA_DURATION_TAG
            print(f"Button {idx}: Failed to extract duration - {e}")

        return duration
    
    def _duration_txt_to_seconds(self, duration_txt):
        if duration_txt == self.UNKNOWN_DURATION_TAG:
            return self.UNKNOWN_DURATION
        elif duration_txt == self.NA_DURATION_TAG:
            return self.NA_DURATION
        
        try:
            minutes, seconds = map(int, duration_txt.split(':'))
            total_seconds = minutes * 60 + seconds
        except:
            total_seconds = self.CONVERSION_FAILED
        return total_seconds
    
    def scrape_audio_URLs(self):

        if os.path.exists(self.audio_urls_file):
            print(f"Audio URLs already scraped and cached at {self.audio_urls_file}.")
            return

        try:
            self.driver.get(self.naxos_url)
            
            # Wait up to 30 seconds for the page to load and content to be ready
            wait = WebDriverWait(self.driver, 30)

            # Wait for the page content to load by checking for either:
            # 1. The "More" button (indicating content is loaded but needs expansion)
            # 2. Play buttons (indicating content is already fully loaded)
            try:
                print("Waiting for page content to load...")
                # First, try to wait for the More button to appear (content partially loaded)
                more_button = wait.until(EC.presence_of_element_located((By.ID, "myMore")))
                print("Page content loaded - 'More' button found.")
                
                # Now wait for it to be clickable and click it
                more_button = wait.until(EC.element_to_be_clickable((By.ID, "myMore")))
                print("'More' button is clickable, clicking to reveal all tracks...")
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", more_button)
                more_button.click()
                print("Clicked 'More' button to reveal additional tracks.")
                
                # Wait for the expanded content to load (wait for more play buttons to appear)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td[id^='divplaystop_'] > a[onclick*='fnPlayStop30']")))
                print("Expanded content loaded successfully.")
                
            except:
                print("No 'More' button found or content already fully loaded.")
                # If no More button, just make sure play buttons are present
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td[id^='divplaystop_'] > a[onclick*='fnPlayStop30']")))
                print("Play buttons are available - content is ready.")


            # Find all play buttons
            time.sleep(2)  # Additional wait to ensure all elements are fully loaded
            play_buttons = self.driver.find_elements(By.CSS_SELECTOR, "td[id^='divplaystop_'] > a[onclick*='fnPlayStop30']")

            print(f"Found {len(play_buttons)} play buttons.")
 
            audio_URLs = []
            durations = []
            titles = []


            for idx, button in enumerate(play_buttons, start=1):
                # Extract the onclick attribute content to parse id and token
                onclick_js = button.get_attribute("onclick")
                if not onclick_js:
                    print(f"Button {idx}: No onclick attribute found.")
                    continue

                # Extract parameters from fnPlayStop30 call: fnPlayStop30('ID', 'TOKEN')
                import re
                match = re.search(r"fnPlayStop30\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", onclick_js)
                if not match:
                    print(f"Button {idx}: Could not parse onclick parameters.")
                    continue
                btn_id, btn_token = match.group(1), match.group(2)

                duration_txt = self._extract_content(button, -1, idx)
                durations.append(self._duration_txt_to_seconds(duration_txt))
                titles.append(self._extract_content(button, -2, idx))

                print(f"\nClicking button {idx} with id: {btn_id}")
                print(f"Token: {btn_token[:60]}...")  # Print first 60 chars of token for brevity

                # Clear performance logs before click
                self.driver.get_log("performance")

                # Click the button
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
                    button.click()
                except Exception as e:
                    print(f"Button {idx}: Click Excepted, skipping.")
                    audio_URLs.append(e.__class__.__name__)
                    continue

                # Wait for the wanted network response (AjxGetAudioUrl) up to a timeout
                found_audio_url = False
                timeout = 10.0  # seconds
                poll_interval = 0.1
                waited = 0.0
                while waited < timeout:
                    logs = self.driver.get_log("performance")
                    for entry in logs:
                        message = entry["message"]
                        if 'Network.responseReceived' in message and 'AjxGetAudioUrl' in message:
                            message_json = json.loads(message)["message"]
                            request_id = message_json["params"]["requestId"]
                            try:
                                response_body = self.driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                                audio_url = response_body.get("body", "")
                                if audio_url:
                                    print(f"Audio URL for button {btn_id}:\n{audio_url}\n")
                                    found_audio_url = True
                                    audio_URLs.append(audio_url)
                                    break
                            except Exception as e:
                                print(f"Failed to get response body: {e}")
                    if found_audio_url:
                        break
                    time.sleep(poll_interval)
                    waited += poll_interval

                if not found_audio_url:
                    # print in red that no audio URL was found
                    print(f"\033[91m /!\\ No audio URL found for button {btn_id}.\033[0m")


        finally:
            self.driver.quit()

        print("All extracted audio URLs:")
        for url in audio_URLs:
            print(f"* {url}")

        # store audio_URLs as a npy file
        np.save(self.audio_urls_file, np.array(audio_URLs))
        np.save(self.audio_full_durations_file, np.array(durations))
        np.save(self.audio_titles_file, np.array(titles))

    def download_audios(self):

        if not os.path.exists(self.audio_urls_file):
            raise FileNotFoundError(f"Audio URLs file not found at {self.audio_urls_file}. Please run scrape_audio_URLs() first.")
        
        if self._is_cached(self.audio_files_path):
            print(f"Audio files already cached in {self.audio_files_path}. Skipping download.")
            return
        
        audio_URLs = np.load(self.audio_urls_file, allow_pickle=True)
        
        # Pad the index with zeros for proper alphabetical sorting (e.g., 001, 002, ... 010, 011, ...)
        num_digits = len(str(len(audio_URLs)))

        for idx, url in enumerate(audio_URLs):
            if isinstance(url, str) and url.startswith("http"):
                response = requests.get(url)
                file_path = os.path.join(self.audio_files_path, f"30s_preview_{str(idx+1).zfill(num_digits)}.mp4")
                with open(file_path, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Skipping invalid URL at index {idx}: {url}")

        mp4_files_in_preview = [f for f in os.listdir(self.audio_files_path) if f.endswith('.mp4')]

        for filename in mp4_files_in_preview:
            audio_file_path = os.path.join(self.audio_files_path, filename)
            sr, ch = self._extract_sr_and_channels_from_mp4(audio_file_path)
            samples = self._extract_samples_from_mp4(audio_file_path, sample_rate=sr, channels=ch)
            print(f'Samples shape: {samples.shape}, Sample rate: {sr}')

            signal = Signal(samples=samples, sample_rate=sr, origine_filename=audio_file_path)
            signal.save_wav(audio_file_path.replace('.mp4', '.wav'))
            # delete the mp4 file to save space
            os.remove(audio_file_path)

    def _extract_samples_from_mp4(self, file_path, sample_rate=44100, channels=2):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_path,
            '-i', file_path,
            '-f', 's16le',         # output raw 16-bit PCM audio
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),  # set audio sampling rate
            '-ac', str(channels),     # number of audio channels
            '-'
        ]

        # Run ffmpeg and capture output bytes
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        raw_audio = process.stdout

        # Convert raw bytes to numpy array of int16 samples
        audio_data = np.frombuffer(raw_audio, dtype=np.int16)

        # Reshape for multi-channel audio
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
            # Average channels to mono (np.mean produces float64, convert back to int16)
            mono_audio = audio_data.mean(axis=1).astype(np.int16)
        else:
            mono_audio = audio_data

        return mono_audio
    
    def _extract_sr_and_channels_from_mp4(self, file_path):
        tag = TinyTag.get(file_path)
        sr, channels = tag.samplerate, tag.channels
        if sr is None or channels is None:
            raise ValueError(f"Could not extract sample rate or channels from {file_path}")
        return sr, channels

    def _clear_cache(self, path):
        if os.path.exists(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    def _is_cached(self, path):
        return os.path.exists(path) and len(os.listdir(path)) > 0
    
    def is_scraping_complete(self):
        contains_audios = self._is_cached(self.audio_files_path)

        contains_meta_files = \
            os.path.exists(self.audio_urls_file) \
        and os.path.exists(self.audio_full_durations_file) \
        and os.path.exists(self.audio_titles_file)

        return contains_audios and contains_meta_files