"""
사번 :
직원명 : 

입력 받고, 로그인 성공시 해당 직원이 관리하는 고객 자료 출력

ex)
고객번호  고객명  고객전화
3       ...     ...
인원수 : x명
"""
import MySQLdb
import pickle
with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)
    
def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        employee_no = input('사원번호 입력 : ')
        employee_name = input('사원명 입력 : ')
        
        sql = "select jikwon_name from jikwon where jikwon_no={0}".format(employee_no)
        
        cursor.execute(sql)
        
        jik_name = cursor.fetchall()
       
        # for jik_name in cursor:
        #     print(jik_name)   # ('사원명',) 이렇게 들어가있음
        
        
        if employee_name == jik_name[0][0]:
            print('로그인 성공')
            
            sql = """
                select gogek_no, gogek_name, gogek_tel from gogek
                where gogek_damsano = {0}
            """.format(employee_no)
            
            cursor.execute(sql)
            gogeks = cursor.fetchall()
            
            print('고객번호 | 고객명 | 고객전화')
            for gogek in gogeks:
                print(gogek[0], '   ', gogek[1],'   ',  gogek[2])
            
            print('인원수 : ', len(gogeks)) 
        else:
            print('로그인 실패')
            print('번호와, 이름을 확인하시오')
            
            
    except Exception as e:
        print('err :', e)
        
    finally:
        cursor.close()
        conn.close()
        
if __name__== '__main__':
    chulbal()    