# 셀리니움 모델로 브라우저 제어

from selenium import webdriver

try:
    url = "https://www.daum.net"
    browser = webdriver.Chrome('D:/work/chromedriver')
    browser.implicitly_wait(3)

    browser.get(url);
    browser.save_screenshot("daum_img.png")
    browser.quit()
    print('성공')
except Exception:
    print('에러')