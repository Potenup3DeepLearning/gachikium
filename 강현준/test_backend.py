from fastapi import FastAPI

app = FastAPI(title='가치키움 API')

@app.get('/')
def read_root():
    return {'message' : '백엔드 서버 정상 작동 중'}

@app.get('/status')
def get_status():    
    return {'total_users' : 10000, 'version': '1.0.0'}