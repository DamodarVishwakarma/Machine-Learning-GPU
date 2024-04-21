import time
import requests
import json
import boto3
import subprocess
import paramiko
from decouple import config


class GpuInstanceHandler:
    def __init__(self):
        self.ec2 = boto3.client(
            'ec2', 
            'us-east-1',
            aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY')
        )
        self.instance_id = "i-0d9f0f039c1f8dd00"
        self.gpu_ip = "52.3.108.30"
        self.gpu_flask_server_url = "http://52.3.108.30:5010"
        self.ssh_key_filepath = "gpt4.pem"
        self.gpu_username = "ubuntu"
        # run flask server as background service in gpu instance
        self.run_flask_command = "nohup /opt/conda/envs/pytorch/bin/python /home/ubuntu/ml_project/pipline_project/4_modeling/modeling_endpoint_api_v5.py > /dev/null 2>&1 &"


    def is_instance_running(self):
        # Fetch the state of the specified EC2 instance.
        response = self.ec2.describe_instances(InstanceIds=[self.instance_id])

        # Extract the instance state.
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
    
        return state == 'running'

    def get_all_instance_ids(self):
        # Fetch the list of EC2 instances. 
        # You might need to handle pagination if you have more than a certain number of instances.
        response = self.ec2.describe_instances()

        instance_ids = []

        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])

        return instance_ids

    def start_instance(self):
        response = self.ec2.start_instances(InstanceIds=[self.instance_id])
        current_state = response['StartingInstances'][0]['CurrentState']['Name']
        
        return current_state

    def stop_instance(self):
        response = self.ec2.stop_instances(InstanceIds=[self.instance_id])
        current_state = response['StoppingInstances'][0]['CurrentState']['Name']
        
        return current_state

    def get_region(self):
        try:
            region = subprocess.check_output(["curl", "-s", "http://169.254.169.254/latest/meta-data/placement/availability-zone"]).decode('utf-8')[:-1]
            return region
        except Exception as e:
            print(f"Error fetching region: {e}")
            return None

    def authorization_message(self):
        sts = boto3.client(
                'sts', 
                'us-east-1',
                aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY')
            )
        decoded_msg = sts.decode_authorization_message(
            EncodedMessage="_hNxvgJH6nSnc3k-bAkJQ_aFGoEr9-xQyXxX-jYiuH6XVid4RMycQnkxzUP_El-3sDakS0NAdfGrRd18U1S19c0m0mVS_L65PW-qrnbjyjG7kMQN0g8M148FPRpldCscE-YZ0wk-94whx17W0PWGK_zJrSKeqIs95HlRCwX89KcmUk1kSRuUbCIMpbV9bGJzh8WUO09IlXkO0dHreU2pxW7thC-DhMRS0R9KlagL4LuNpbE8ivdlBxevMPHRyf9U6-S-E-Vwo-cZrYVA3Ow1-8acoCsNMv-XgHkvu6tfpOD4ilfrim0TKvWCgoOXNHv3wJjCSQLMw6bmGB7B5PRn3EI6jDkutBma3myeHUPHyUinkHFJyr-BAzup-Urr0LRYzti9MVefTDNMgJApg7DLMYjGZuGxJcR8p0kR-589lpexfrW9GEK_6SXlRpU9g4RtmV45956nh5K1pTUwxsNqJ5iDHbzfKq-m2f4mFSqYSWiL8NC5avXNvEV6XmtpxYUTa76q4NF9h-YTJCn9KCUqJy_GcJVN5xezd2EmvVx9xoQGoZQaBHN_q2JHHLlJAP6bXcWe8gaZLILd4mAr-_nut26vp5Eu_Rq2F_Vk-xkJlydgluAzqmbvuISbdBnwVt8wIu134albwCGcctGYOVDcqCVT_nsU5KifcofgitbOtkMG-1ygqEHZoIUlTEJasymUfoDYYNdgYJ964fS2YkosoNgqCClG6oOxKDHSu3YIHCSzQK7frzoFBS49weJrv3lZ_06c")  # Put the encoded message here
        print(decoded_msg['DecodedMessage'])

    def execute_flask_on_gpu(self):
        """
        Execute a command on a remote GPU server and return the output.
        """
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh_client.connect(
                self.gpu_ip, username=self.gpu_username, key_filename=self.ssh_key_filepath)
        except paramiko.SSHException as e:
            # Handle the exception safely and inform the user.
            print("Error connecting to the GPU server:", e)
            return None

        # Execute the command
        try:
            stdin, stdout, stderr = ssh_client.exec_command(self.run_flask_command)
            result = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            print("=========", error)
        except paramiko.SSHException as e:
            # Handle the exception safely and inform the user.
            print("Error executing command on the GPU server:", e)
            return None
        
        # Close the connection
        ssh_client.close()
        
        return result

        # Example usage
        output = execute_on_gpu('gpu_instance_ip', '/path_to_private_key.pem', 'python your_gpu_script.py')
        print(output)


    def check_flask_server_running(self):
        """
        Check flask is running on gpu instance
        """
        status = "not running"
        try:
            response = requests.get(f'{self.gpu_flask_server_url}/check_flask_server_status')
            data = response.json()
            status = data["message"]
        except:
            pass
        if (status == "running"):
            print("Gpu instance is running...")
        return status == "running"

    def send_api_request_to_gpu(self, data):
        """
        Send post request to gpu instance and get response
        """
        print("start to send a post request to gpu instance ...")
        headers = {
            'Content-Type': 'application/json'
        }
        try:
            response = requests.post(
                f'{self.gpu_flask_server_url}/upload_data_train_model',
                data=json.dumps(data),
                headers=headers
            )
            data = response.json()
            status = data["message"]
            print(f"gpu instance response: {status}")
            return data
        except Exception as err:
            print(f"gpu error: {err}")

        return {"message": "gpu error"}
    
    def run_on_gpu(self, data):
        """
        Main processing for gpu instance
        """
        print("start to run gpu instance...")
        # check gpu instance is running and run gpu instance if it is not running
        if not self.is_instance_running():
            self.start_instance()

        # it will take some time to turn on gpu instance
        while not self.is_instance_running():
            time.sleep(1)

        # check if flask server is running in gpu instance
        if not self.check_flask_server_running():
            self.execute_flask_on_gpu()

        while not self.check_flask_server_running():
            time.sleep(1)

        # train data in gpu instance
        response = self.send_api_request_to_gpu(data)
        return response

# instance_handler = GpuInstanceHandler()
# state = instance_handler.execute_flask_on_gpu()
# print("==== ", state)



