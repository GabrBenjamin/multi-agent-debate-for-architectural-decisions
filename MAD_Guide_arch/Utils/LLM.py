import paramiko
import json
from langchain.llms import BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult, Generation
from pydantic import PrivateAttr
import shlex

# Define the SSHConnection class
class SSHConnection:
    def __init__(self):
        self.hostname = '10.27.0.15'
        self.username = 'felipe'
        self.password = 'hu9e2on32093d0j929cu0id2'
        self.local_port = 11434  
        self.remote_port = 11434
        self.ssh = paramiko.SSHClient()
        self.channel = None
        
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            self.ssh.connect(self.hostname, username=self.username, password=self.password)
        except (paramiko.AuthenticationException, paramiko.SSHException, Exception) as e:
            print(f"Error connecting: {e}")
    
    def send_api_request(self, data):
        json_data = json.dumps(data)
        quoted_json_data = shlex.quote(json_data)  # Escape JSON string
        command = f"curl -X POST http://localhost:{self.remote_port}/api/chat -d {quoted_json_data} -H 'Content-Type: application/json'"
        print(f"Executing command: {command}")  # Debug: Print the command
        try:
            stdin, stdout, stderr = self.ssh.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                print(f"Error from server: {error}")
                return None
            return json.loads(output)  # Parse the JSON response
        except paramiko.SSHException as e:
            print(f"Error executing command: {e}")
            return None

    def close(self):
            self.ssh.close()


class LlamaSSHLLM(BaseLLM):
    _ssh_connection: SSHConnection = PrivateAttr()  # Mark ssh_connection as a private attribute

    def __init__(self):
        super().__init__()  # Call the parent class initializer
        self._ssh_connection = SSHConnection()  # Initialize the SSH connection

    def _call(self, prompt: str, stop: list = None) -> str:
        """Handles simple calls for internal usage."""
        request_data = {
            "model": "llama3.3",  # Replace with your model name
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},  # Adjust as needed
            "prompt_eval_count": 512,  # Adjust max tokens as needed
            "stream": False,
        }
        response_data = self._ssh_connection.send_api_request(request_data)
        if not response_data:
            raise ValueError("No response from the Llama server.")
        content = response_data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("No content returned in the response.")
        return content

    def _generate(self, prompts, stop=None):
        """Implements LangChain's abstract `_generate` method."""
        generations = []
        for prompt in prompts:
            output_text = self._call(prompt, stop)
            generations.append(Generation(text=output_text))
        return LLMResult(generations=[generations])

    @property
    def _llm_type(self):
        return "llama_ssh"

    def close(self):
        """Closes the SSH connection."""
        self._ssh_connection.close()