import os
import time
import random
import requests
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager


def random_delay(min_seconds=1, max_seconds=2):
    """Add random delay to mimic human behavior"""
    time.sleep(random.uniform(min_seconds, max_seconds))


def human_like_typing(element, text):
    """Type text with random delays like a human"""
    element.clear()
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.02, 0.1))


def setup_stealth_driver():
    """Set up Chrome with maximum stealth options"""
    options = Options()

    # Basic stealth options
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # Advanced anti-detection
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins-discovery")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # Fresh profile
    options.add_argument("--user-data-dir=/tmp/chrome_stealth_profile")

    # Window size
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)

    # Execute stealth script
    stealth_script = """
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined,
    });
    
    Object.defineProperty(navigator, 'plugins', {
      get: () => [1, 2, 3, 4, 5],
    });
    
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });
    
    window.chrome = {
      runtime: {},
    };
    
    Object.defineProperty(navigator, 'permissions', {
      get: () => ({
        query: () => Promise.resolve({ state: 'granted' }),
      }),
    });
    """
    driver.execute_script(stealth_script)

    return driver


# ===== Main Script =====
SEARCH_QUERY = "bus service smrt and sbs bus front view on road"
NUM_IMAGES = 300
FOLDER_NAME = "img/scraped_bus_images"

print("[+] Setting up stealth browser...")
driver = setup_stealth_driver()

try:
    # Go directly to Google Images
    print("[+] Opening Google Images...")
    driver.get("https://images.google.com/")
    random_delay(1, 3)

    # Search with human-like behavior
    print(f"[+] Searching for: {SEARCH_QUERY}")
    search_box = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )

    # Click search box and type like human
    actions = ActionChains(driver)
    actions.move_to_element(search_box).click().perform()
    random_delay(0.5, 1)

    human_like_typing(search_box, SEARCH_QUERY)
    random_delay(0.5, 1)
    search_box.send_keys(Keys.RETURN)

    print("[+] Waiting for results to load...")
    random_delay(2, 5)

    # Debug: Check what we have on the page
    print(f"[DEBUG] Current URL: {driver.current_url}")
    print(f"[DEBUG] Page title: {driver.title}")

    # Wait for images to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img"))
        )
        print("[+] Images detected on page")
    except:
        print("[!] No images found on page")

    # Try multiple approaches to find image thumbnails
    all_selectors = [
        # Current Google Images selectors (2024)
        "div[data-ri] img",
        "img[data-src*='tbn']",
        "img.rg_i",
        "img.Q4LuWd",
        "img[jsname]",
        "a[jsname] img",
        ".islrc img",
        ".rg_bx img",
        ".isv-r img",
        # Generic selectors
        "img[alt]:not([alt=''])",
        "img[src*='gstatic']",
        "img[src*='encrypted-tbn']"
    ]

    print(f"\n[+] Testing {len(all_selectors)} different selectors...")

    best_thumbnails = []
    best_selector = None

    for selector in all_selectors:
        try:
            thumbnails = driver.find_elements(By.CSS_SELECTOR, selector)
            # Filter out small images (likely UI elements)
            valid_thumbnails = []
            for thumb in thumbnails:
                try:
                    size = thumb.size
                    if size['width'] > 50 and size['height'] > 50:
                        valid_thumbnails.append(thumb)
                except:
                    continue

            print(
                f"[DEBUG] '{selector}': {len(valid_thumbnails)} valid thumbnails")

            if len(valid_thumbnails) > len(best_thumbnails):
                best_thumbnails = valid_thumbnails
                best_selector = selector

        except Exception as e:
            print(f"[DEBUG] Error with '{selector}': {str(e)[:30]}")

    if not best_thumbnails:
        print("[!] No image thumbnails found!")
        print("[!] Possible issues:")
        print("    - Google changed their structure")
        print("    - Search returned no results")
        print("    - Page not fully loaded")
        print("    - Being blocked by Google")

        # Save page source for debugging
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("[DEBUG] Page source saved to 'debug_page.html'")

        # Try to take a screenshot
        try:
            driver.save_screenshot("debug_screenshot.png")
            print("[DEBUG] Screenshot saved to 'debug_screenshot.png'")
        except:
            pass

        exit()

    print(
        f"\n[+] Found {len(best_thumbnails)} thumbnails using selector: '{best_selector}'")
    print(
        f"[+] Processing first {min(NUM_IMAGES, len(best_thumbnails))} images...")

    # Collect images
    image_urls = set()
    processed = 0

    for i, thumb in enumerate(best_thumbnails[:NUM_IMAGES]):
        try:
            print(
                f"\n[+] Processing thumbnail {i+1}/{min(NUM_IMAGES, len(best_thumbnails))}")

            # Scroll thumbnail into view
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", thumb)
            random_delay(1, 2)

            # Get thumbnail info before clicking
            thumb_src = thumb.get_attribute(
                "src") or thumb.get_attribute("data-src")
            print(
                f"    Thumbnail src: {thumb_src[:50] if thumb_src else 'None'}...")

            # Click thumbnail with retry
            click_success = False
            for attempt in range(3):
                try:
                    # Try different click methods
                    if attempt == 0:
                        # Normal click
                        thumb.click()
                    elif attempt == 1:
                        # JavaScript click
                        driver.execute_script("arguments[0].click();", thumb)
                    else:
                        # ActionChains click
                        actions = ActionChains(driver)
                        actions.move_to_element(thumb).click().perform()

                    click_success = True
                    print(f"    [✓] Clicked thumbnail (attempt {attempt + 1})")
                    break

                except Exception as click_error:
                    print(
                        f"    [!] Click attempt {attempt + 1} failed: {str(click_error)[:30]}")
                    random_delay(0.5, 1)

            if not click_success:
                print(f"    [!] All click attempts failed")
                continue

            # random_delay(0.5, 1)

            # Look for expanded/full-size image with multiple selectors
            expanded_selectors = [
                "img.n3VNCb",           # Most common
                "img[data-noaft='1']",  # Alternative
                "img.sFlh5c",           # Another variant
                "img.iPVvYb",           # Yet another
                "img[jsname='HiaYvf']",  # With jsname
                "img[jsname='kn3ccd']",  # Different jsname
                ".islsp img",           # In side panel
                ".v4dQwb img",          # In viewer
                "img[style*='max-width']",  # Styled image
                # Any http image not thumbnail
                "img[src*='http']:not([src*='tbn'])"
            ]

            expanded_img = None
            used_selector = None

            for exp_selector in expanded_selectors:
                try:
                    potential_imgs = driver.find_elements(
                        By.CSS_SELECTOR, exp_selector)
                    for img in potential_imgs:
                        # Check if image is actually displayed and large enough
                        if img.is_displayed():
                            size = img.size
                            if size['width'] > 200 and size['height'] > 200:
                                expanded_img = img
                                used_selector = exp_selector
                                break
                    if expanded_img:
                        break
                except:
                    continue

            if expanded_img:
                img_url = expanded_img.get_attribute("src")
                print(img_url)
                print(f"    [✓] Found expanded image with: {used_selector}")
                print(f"    URL: {img_url[:60] if img_url else 'None'}...")
                if (
                    img_url
                    and img_url.startswith("http")
                    and "data:image" not in img_url
                    and "tbn" not in img_url
                ):
                    image_urls.add(img_url)
                    processed += 1
                    print(f"    [✓] Added to collection ({processed} total)")
                else:
                    print(f"    [!] Invalid or thumbnail URL")
            else:
                print(f"    [!] No expanded image found")

            # Small random pause
            # random_delay(0.5, 2)

        except Exception as e:
            print(f"    [!] Error processing thumbnail {i+1}: {str(e)[:50]}")
            continue

    print(
        f"\n[+] Collection complete! Found {len(image_urls)} unique image URLs")

    # Download images
    if image_urls:
        os.makedirs(FOLDER_NAME, exist_ok=True)
        downloaded = 0

        for i, url in enumerate(list(image_urls)):
            try:
                print(f"[+] Downloading image {i+1}/{len(image_urls)}...")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://images.google.com/',
                }

                response = requests.get(url, timeout=15, headers=headers)
                response.raise_for_status()

                # Check content type and only proceed with JPG/JPEG/PNG
                content_type = response.headers.get('content-type', '').lower()

                # Define allowed image types
                allowed_types = ['image/jpeg', 'image/jpg', 'image/png']

                # Check if the content type is allowed
                if not any(allowed_type in content_type for allowed_type in allowed_types):
                    print(
                        f"    [!] Skipping unsupported format: {content_type}")
                    continue

                # Determine file extension based on content type
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                elif 'png' in content_type:
                    ext = 'png'
                else:
                    # Skip if we can't determine a valid extension
                    print(
                        f"    [!] Skipping unknown image type: {content_type}")
                    continue

                # Additional validation: Check file size (optional)
                if len(response.content) < 1024:  # Skip files smaller than 1KB
                    print(
                        f"    [!] Skipping too small file ({len(response.content)} bytes)")
                    continue

                # Check magic bytes for additional validation
                content_start = response.content[:10]
                is_valid_image = False

                # JPEG magic bytes
                if content_start.startswith(b'\xff\xd8\xff'):
                    is_valid_image = True
                    ext = 'jpg'  # Ensure correct extension
                # PNG magic bytes
                elif content_start.startswith(b'\x89PNG\r\n\x1a\n'):
                    is_valid_image = True
                    ext = 'png'  # Ensure correct extension

                if not is_valid_image:
                    print(f"    [!] Invalid image format detected, skipping")
                    continue

                file_path = os.path.join(
                    FOLDER_NAME, f"bus_{downloaded+1}.{ext}")

                with open(file_path, "wb") as f:
                    f.write(response.content)

                downloaded += 1
                print(
                    f"    [✓] Saved: {file_path} ({ext.upper()}, {len(response.content)} bytes)")

                # Small delay between downloads
                random_delay(1, 3)

            except Exception as e:
                print(f"    [!] Download failed: {str(e)[:50]}")

        print(
            f"\n✅ Success! Downloaded {downloaded} JPG/PNG images to '{FOLDER_NAME}' folder")
    else:
        print("[!] No images were collected to download")

except Exception as e:
    print(f"[!] Script error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n[+] Closing browser...")
    random_delay(1, 2)
    driver.quit()
