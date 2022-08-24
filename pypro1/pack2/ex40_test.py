import MySQLdb
import pickle
with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)
    
def output():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        bunho = input('사번:')
        erum = input('직원명:')
        
        sql = """
            select jikwon_name 
            from jikwon
            where jikwon_no={0}
            """.format(bunho)
        cursor.execute(sql)
        
        employee_name = cursor.fetchone()
        print(employee_name[0])
        
        if erum == employee_name[0]:
            print('로그인 하였습니다')
            
            sql = """
                select gogek_no, gogek_name, gogek_tel from gogek
                where gogek_damsano = {0}
            """.format(bunho)
            
            cursor.execute(sql)
            customers = cursor.fetchall()
            
            print('고객번호\t고객명\t고객전화')    
            for gogek in customers:
                print(gogek[0], '\t' , gogek[1], '\t', gogek[2])
                
            print('고객수: '+ str(len(customers)))
        
        else:
            print('로그인 실패')
            print('사번과 이름을 확인하세요')
    
    except Exception as e:
        print('err:', e)
    finally:
        cursor.close()
        conn.close()
        
if __name__=='__main__':
    output()