# 예외처리 : 프로그램 진행 도중 발생되는 에러에 대한 처리 방법 중 하나
# try ~ except

def divide(a, b):
    return a / b

# c = divide(5, 2)
# c = divide(5, 0)  # ZeroDivisionError : division by zero
# print('c:',c)

try:
    c = divide(5, 2)    # try 블럭 내에 에러가 없으면 블럭 내용을 수행하고 종료됨.
    # c = divide(5, 0)  # try 블럭 내에 에러가 있으면 except 내용을 수행함.
    print('c:',c)
    
    mbc = [1, 2]
    # print(mbc[0])  # 참조범위 오류: list index out of range
    print(mbc[2])
    # f = open('c:/work/aa.txt')  # 기타 나머지 에러: [Errno 2] No such file or directory: 'c:/work/aa.txt'
    
except ZeroDivisionError:
    print('두번째 인자로 0을 주지 마세요.')

except IndexError as err:
    print('참조범위 오류:', err)  # 시스템이 제공하는 에러메세지도 같이 보기
    
except Exception as e:
    print('기타 나머지 에러:', e)

finally:
    print('에러 유무와 상관없이 반드시 수행되는 부분')
    
print('프로그램 종료')