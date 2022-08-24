import MySQLdb
import pickle
with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)
    
def output():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
    
        jik = input('직급 입력:')
    
        sql = """
        select jikwon_no, jikwon_name, jikwon_jik, buser_num 
        from jikwon 
        where jikwon_jik="{0}"
        """.format(jik)
        cursor.execute(sql)
        
        datas = cursor.fetchall()
        for data in datas:
            print(data[0], data[1], data[2], data[3])
        
    except Exception as e:
        print('err:', e)
    finally:
        cursor.close()
        conn.close()
    
if __name__=='__main__':
    output()  
        