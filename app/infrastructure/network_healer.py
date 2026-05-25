import socket
import logging
import asyncio
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class NetworkAutoHealer:
    def __init__(self, check_host: str = "api.groq.com", check_port: int = 443, timeout: int = 3):
        self.check_host = check_host
        self.check_port = check_port
        self.timeout = timeout

    def check_connection(self) -> bool:
        """
        Quick check if the internet connection to check_host is active.
        """
        try:
            socket.setdefaulttimeout(self.timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.check_host, self.check_port))
            return True
        except Exception:
            return False

    async def heal(self) -> bool:
        """
        Playwright automation script to bypass captive portals and re-establish internet access.
        """
        logger.info("Starting captive portal auto-healing bypass...")
        
        # Test connection first; if it's already active, skip bypass
        if self.check_connection():
            logger.info("Internet connection is healthy. No healing needed.")
            return True

        async with async_playwright() as p:
            try:
                # Launch headless browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Navigate to neverssl.com to trigger captive portal redirection
                logger.info("Navigating to neverssl.com to detect captive portal...")
                await page.goto("http://neverssl.com", timeout=10000)
                
                # Wait for redirection to settle
                await page.wait_for_timeout(2000)
                
                # Common captive portal button and link texts in VN and EN
                selectors = [
                    "button:has-text('Tiếp tục')", 
                    "button:has-text('Connect')", 
                    "button:has-text('Đồng ý')", 
                    "a:has-text('Connect')", 
                    "button:has-text('Access')", 
                    "input[type='submit']",
                    "button:has-text('Bắt đầu')",
                    "button:has-text('Internet')"
                ]
                
                clicked = False
                for sel in selectors:
                    try:
                        element = page.locator(sel)
                        if await element.count() > 0:
                            logger.info(f"Found captive portal control: {sel}. Clicking...")
                            await element.first.click()
                            clicked = True
                            await page.wait_for_timeout(3000)
                            break
                    except Exception as click_err:
                        logger.warning(f"Could not click portal element {sel}: {click_err}")

                if not clicked:
                    logger.info("No obvious captive portal buttons found. Captive portal might be clear or unknown.")
                
                # Verify if connectivity is restored
                await browser.close()
                
                # Retest connection
                for i in range(3):
                    await asyncio.sleep(2)
                    if self.check_connection():
                        logger.info("Connectivity successfully restored after healing cycle.")
                        return True
                
                return False
                
            except Exception as e:
                logger.error(f"Captive portal healing script encountered an error: {e}")
                return False
