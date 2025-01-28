import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import os
from datetime import datetime
from urllib.parse import quote_plus
import hashlib
from PIL import Image

class EbayListingParser:
    """Handles the parsing of individual eBay listings and their details"""
    
    def __init__(self):
        self.description_cache = {}
    
    def parse_listing(self, soup_item, download_images=False, images_dir=None):
        item_data = {}
        
        try:
            # Core fields
            item_data['title'] = soup_item.find('span', role='heading').get_text(strip=True)
            item_data['price'] = self._clean_price(soup_item.find('span', class_='s-item__price'))
            item_data['condition'] = self._get_text(soup_item.find('span', class_='s-item__condition'))
            item_data['shipping'] = self._clean_shipping(soup_item.find('span', class_='s-item__shipping'))
            item_data['url'] = soup_item.find('a', class_='s-item__link')['href'].split('?')[0]

            # Dynamic details from search listing
            details = soup_item.find_all('div', class_='s-item__detail')
            for detail in details:
                label = detail.find('span', class_='s-item__label')
                value = detail.find('span', class_='s-item__dynamic')
                if label and value:
                    key = self._clean_key(label.get_text(strip=True))
                    item_data[key] = value.get_text(strip=True)

            # Time and sales info
            item_data['listing_type'] = 'sold' if soup_item.find('span', class_='s-item__end-date') else 'active'
            item_data['end_time'] = self._get_text(soup_item.find('span', class_='s-item__end-date'))

            # Get detailed description
            details = self.scrape_item_details(item_data['url'])
            item_data.update(details)

            # Handle image downloading if requested
            if download_images and 'images' in details and details['images']:
                folder_name = self._sanitize_folder_name(item_data['title']) + "-photos"
                save_dir = os.path.join(images_dir, folder_name)
                unique_images = self._download_images(details['images'], save_dir)
                item_data['image_folder'] = os.path.relpath(save_dir, os.path.dirname(images_dir))
                item_data['full_filepath'] = os.path.abspath(save_dir)
                item_data['images'] = unique_images  # Update with unique images
            else:
                item_data['image_folder'] = None
                item_data['full_filepath'] = None

            return item_data
                
        except Exception as e:
            print(f"Error parsing listing: {str(e)[:50]}")
            return None

    def scrape_item_details(self, url):
        """Scrape detailed description and photos from product page"""
        if url in self.description_cache:
            return self.description_cache[url]

        details = {}
        html = self._fetch_page(url)
        if not html:
            return details

        try:
            soup = BeautifulSoup(html, 'lxml')

            # Scrape item specifics
            specifics_section = soup.find('div', {'class': 'ux-layout-section-evo__item'})
            if specifics_section:
                for row in specifics_section.find_all('div', {'class': 'ux-layout-section-evo__row'}):
                    label = row.find('dt').get_text(strip=True)
                    value = row.find('dd').get_text(strip=True).replace('\xa0', ' ')
                    key = self._clean_key(label)
                    details[key] = value

            # Scrape full description
            description_section = soup.find('div', {'class': 'd-item-description'})
            if description_section:
                details['full_description'] = description_section.get_text(' ', strip=True)
            else:
                details['full_description'] = None

            # Get image URLs
            image_tags = soup.find_all('img')
            image_urls = []
            for img in image_tags:
                image_url = img.get('data-zoom-src') or img.get('src')
                if image_url and image_url.startswith('http'):
                    image_url = re.split(r'\?', image_url)[0]
                    image_urls.append(image_url)
            
            details['images'] = image_urls

            self.description_cache[url] = details
            return details

        except Exception as e:
            print(f"Error scraping details: {str(e)[:50]}")
            return details

    # Helper methods
    def _fetch_page(self, url, max_retries=3):
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            ]),
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.ebay.com/',
            'DNT': '1'
        }
        
        session = requests.Session()
        for attempt in range(max_retries):
            try:
                response = session.get(url, headers=headers, timeout=(3, 10))
                response.raise_for_status()
                
                if "sign in to check your access" in response.text.lower():
                    print(f"Blocked by eBay on attempt {attempt+1}")
                    time.sleep(2 ** attempt)
                    continue
                    
                return response.text
                
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)[:50]}")
                time.sleep(random.uniform(1, 3))
        return None

    def _clean_price(self, price_tag):
        if price_tag:
            price_text = price_tag.get_text(strip=True).replace('$', '').replace(',', '')
            match = re.search(r'\d+\.?\d*', price_text)
            return float(match.group()) if match else None
        return None

    def _clean_shipping(self, shipping_tag):
        if shipping_tag:
            text = shipping_tag.get_text(strip=True).lower()
            if 'free' in text:
                return 0.0
            numbers = re.findall(r'\d+\.?\d*', text)
            return float(numbers[0]) if numbers else None
        return None

    def _clean_key(self, key):
        key = re.sub(r'[^\w\s-]', '', key.strip().lower())
        return re.sub(r'[\s-]+', '_', key).strip('_')

    def _get_text(self, element):
        return element.get_text(strip=True) if element else None

    def _sanitize_folder_name(self, name):
        sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', name)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized[:255]

    def _download_images(self, image_urls, save_dir):
        """Download images that meet minimum size requirements and remove duplicates within the item
        Args:
            image_urls (list): List of image URLs to download
            save_dir (str): Directory to save the images
        Returns:
            list: List of unique image file paths
        """
        os.makedirs(save_dir, exist_ok=True)
        session = requests.Session()
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            ]),
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.ebay.com/',
            'DNT': '1'
        }
        
        MIN_WIDTH = 200
        MIN_HEIGHT = 200
        unique_hashes = set()
        unique_images = []

        for i, img_url in enumerate(image_urls):
            temp_path = None
            try:
                response = session.get(img_url, headers=headers, stream=True, timeout=(5, 15))
                response.raise_for_status()
                
                # Create a temporary file with a unique name
                temp_path = os.path.join(save_dir, f'temp_{i}_{int(time.time()*1000)}.jpg')
                
                # Download the image
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                
                # Check image dimensions and format
                img = Image.open(temp_path)
                width, height = img.size
                img_format = img.format.lower() if img.format else 'jpg'
                img.close()  # Important: close the image after checking
                
                # Skip if too small
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    print(f"Skipping image {i + 1} - too small ({width}x{height})")
                    os.remove(temp_path)
                    continue
                
                # Compute SHA256 hash of the image bytes for exact duplicate detection
                with open(temp_path, 'rb') as f:
                    img_bytes = f.read()
                    img_hash = hashlib.sha256(img_bytes).hexdigest()
                
                if img_hash in unique_hashes:
                    print(f"Duplicate image detected and removed: {img_url}")
                    os.remove(temp_path)
                    continue
                else:
                    unique_hashes.add(img_hash)
                
                # Determine final extension and path
                if img_format not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    img_format = 'jpg'
                final_path = os.path.join(save_dir, f'image_{i + 1}.{img_format}')
                
                # Remove existing file if it exists
                if os.path.exists(final_path):
                    os.remove(final_path)
                    
                # Try to rename with exponential backoff
                for attempt in range(3):
                    try:
                        os.rename(temp_path, final_path)
                        print(f"Downloaded image {i + 1} ({width}x{height}) to {final_path}")
                        unique_images.append(final_path)
                        break
                    except OSError as e:
                        if attempt == 2:  # Last attempt
                            raise
                        time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass  # Ignore errors in cleanup
                
            # Small delay between downloads
            time.sleep(0.5)
        
        return unique_images

class EbayScraper:
    """Handles the high-level scraping operations and data management"""
    
    def __init__(self, data_dir="ebay_data"):
        self.DATA_DIR = data_dir
        self._create_data_dir()
        self.parser = EbayListingParser()

    def _create_data_dir(self):
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

    def scrape_search(self, query, max_items=50, pages=None, sold_items=False, download_images=True, query_folder=None):
        items = []
        page_num = 1
        base_url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(query)}"
        
        if sold_items:
            base_url += "&LH_Sold=1&LH_Complete=1"

        # Set up images directory if needed
        images_dir = os.path.join(query_folder, "images") if download_images else None
        if images_dir:
            os.makedirs(images_dir, exist_ok=True)

        while len(items) < max_items and (not pages or page_num <= pages):
            search_url = f"{base_url}&_pgn={page_num}"
            print(f"Scraping page {page_num} - {search_url}")

            html = self.parser._fetch_page(search_url)
            if not html:
                break

            soup = BeautifulSoup(html, 'lxml')
            listings = soup.find_all('div', class_='s-item__wrapper')
            
            if not listings:
                print("No listings found - potential blocking or end of results")
                break

            for listing in listings:
                if len(items) >= max_items:
                    break
                
                parsed = self.parser.parse_listing(listing, download_images, images_dir)
                if parsed:
                    parsed['query'] = query
                    parsed['scraped_at'] = datetime.now().isoformat()
                    items.append(parsed)
                    print(f"Collected {len(items)}/{max_items} - {parsed['title'][:50]}...")

            page_num += 1
            time.sleep(random.uniform(1, 3))

        return items

    def save_to_csv(self, items, query_folder):
        if not os.path.exists(query_folder):
            os.makedirs(query_folder)
    
        if not items:
            print("No data to save")
            return None

        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"ebay_listings_{date_str}.csv"
        filepath = os.path.join(query_folder, filename)

        # Collect all possible columns
        all_keys = set()
        for item in items:
            all_keys.update(item.keys())

        # Prioritize common fields
        column_order = [
            'title', 'price', 'condition', 'shipping', 'url', 
            'listing_type', 'end_time', 'query', 'scraped_at',
            'full_description', 'image_folder', 'full_filepath'
        ]
        remaining_keys = sorted(all_keys - set(column_order))
        columns = column_order + remaining_keys

        df = pd.DataFrame(items, columns=columns)

        # Clean numeric columns
        numeric_cols = ['price', 'shipping'] + [col for col in df.columns if 'price' in col.lower()]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Saved {len(df)} items to {filepath}")
        return df

    def remove_duplicate_images_globally(self, directory=None):
        """Optional: Remove duplicate images across all items based on SHA256 hash
        Args:
            directory (str, optional): Specific directory to check. If None, checks entire data directory.
        """
        check_dir = directory if directory else self.DATA_DIR
        print(f"\nChecking for duplicate images in: {check_dir}")
        
        hash_set = {}
        duplicates = 0
        
        for root, dirs, files in os.walk(check_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()
                        
                        if file_hash in hash_set:
                            os.remove(file_path)
                            duplicates += 1
                            print(f"Deleted duplicate image: {file_path}")
                        else:
                            hash_set[file_hash] = file_path
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        
        print(f"Duplicate removal completed. {duplicates} duplicates deleted.\n")


def main():
    # Initialize the scraper
    scraper = EbayScraper()
    
    # List of search queries with their configurations
    search_queries = [
        {
            "query": "Samsung Galaxy Tab S8 Ultra",
            "max_items": 50,
            "pages": 10,
            "download_images": True,
            "sold_items": True
        },
        {
            "query": "Apple iPad Pro M1 12.9 2021",
            "max_items": 50,
            "pages": 10,
            "download_images": True,
            "sold_items": True
        },
        {
            "query": "Microsoft Surface Pro 8",
            "max_items": 50,
            "pages": 10,
            "download_images": True,
            "sold_items": True
        },
        {
            "query": "Lenovo Tab P12 Pro",
            "max_items": 50,
            "pages": 10,
            "download_images": True,
            "sold_items": True
        },
        {
            "query": "Huawei MatePad Pro 12.6",
            "max_items": 50,
            "pages": 10,
            "download_images": True,
            "sold_items": True
        },
    ]

    all_items = []
    
    for sq in search_queries:
        query = sq.get("query")
        max_items = sq.get("max_items", 50)
        pages = sq.get("pages")
        download_images = sq.get("download_images", True)
        sold_items = sq.get("sold_items", False)

        # Create query-specific folder
        sanitized_query = scraper.parser._sanitize_folder_name(query)
        query_folder = os.path.join(scraper.DATA_DIR, sanitized_query)
        os.makedirs(query_folder, exist_ok=True)
        
        print(f"\n{'='*50}\nScraping query: {query}\n{'='*50}")
        
        # Scrape items for this query
        items = scraper.scrape_search(
            query=query,
            max_items=max_items,
            pages=pages,
            sold_items=sold_items,
            download_images=download_images,
            query_folder=query_folder
        )
        
        if items:
            # Save individual query results
            scraper.save_to_csv(items, query_folder)
            
            # Check for and remove duplicates in this query's image folder
            # Note: Duplicates within each item's images are already handled during download
            # If you still want to perform a global duplicate check, uncomment the following line
            # scraper.remove_duplicate_images_globally(query_folder)
                
            all_items.extend(items)
        else:
            print(f"No items collected for query: {query}")
    
    if all_items:
        # Save master CSV with all results
        master_csv_path = os.path.join(scraper.DATA_DIR, f"ebay_listings_master_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Get all possible columns
        all_keys = set()
        for item in all_items:
            all_keys.update(item.keys())
            
        # Create sorted column list with priority fields
        column_order = [
            'title', 'price', 'condition', 'shipping', 'url', 
            'listing_type', 'end_time', 'query', 'scraped_at',
            'full_description', 'image_folder', 'full_filepath'
        ]
        remaining_keys = sorted(all_keys - set(column_order))
        columns = column_order + remaining_keys
        
        # Create and clean master DataFrame
        df_master = pd.DataFrame(all_items, columns=columns)
        numeric_cols = ['price', 'shipping'] + [col for col in df_master.columns if 'price' in col.lower()]
        for col in numeric_cols:
            if col in df_master.columns:
                df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
                
        # Save master CSV
        df_master.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved master CSV with {len(df_master)} items to {master_csv_path}")
    else:
        print("No items collected from any queries")
    
    # Optional: Final duplicate check across all folders (if desired)
    # print("Performing final duplicate check across all folders...")
    # scraper.remove_duplicate_images_globally()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Runtime: {time.time() - start_time:.2f} seconds")
