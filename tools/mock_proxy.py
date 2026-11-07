# Заглушка LLM-прокси: скриптует tool-calls, чтобы проверить цикл агента, учёт токенов и проверку тестов.
import json, http.server
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        m=b['messages']; n=sum(1 for x in m if x['role']=='tool')
        instr=m[1]['content']
        def call(name,args): return {"role":"assistant","content":"","tool_calls":[{"id":f"c{n}","type":"function","function":{"name":name,"arguments":json.dumps(args)}}]}
        if 'test_mylib' in instr:
            # шаг0: сразу finish без починки (должен быть отклонён), шаг1: чинит, шаг2: finish
            if n==0: msg=call("finish",{"answer":"DONE"})
            elif n==1: msg=call("bash",{"cmd":"sed -i 's/x < best/x > best/; s/% 2 == 1/% 2 == 0/' mylib.py"})
            else: msg=call("finish",{"answer":"DONE"})
        else:
            msg=call("finish",{"answer":"42"})
        out={"choices":[{"message":msg}],"usage":{"prompt_tokens":500,"completion_tokens":40}}
        d=json.dumps(out).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(d)
http.server.HTTPServer(('127.0.0.1',8099),H).serve_forever()
