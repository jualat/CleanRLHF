import json
import logging
import os
import socket
import subprocess
import sys


def start_feedback_server(port):
    """
    Start the feedback server as a separate process/thread.
    Example:
    --------
    server_process = start_feedback_server()
    """
    if is_port_available(int(port)):
        logging.info(f"Starting feedback server on port {port}...")
    else:
        logging.error(f"Port {port} is already in use. Please choose a different port.")
    server_script = os.path.abspath("../humanFeedback/feedback_server.py")
    server_process = subprocess.Popen(
        [sys.executable, server_script, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logging.info("Feedback server started.")
    return server_process


def stop_feedback_server(server_process):
    """
    Stop the feedback server gracefully.
    Example:
    --------
    stop_feedback_server(server_process)
    """
    logging.info("Stopping feedback server...")
    server_process.terminate()
    server_process.wait()
    logging.info("Feedback server stopped.")


def save_feedback_to_file(feedback_data, filename="feedback.json"):
    with open(filename, "w") as f:
        json.dump(feedback_data, f)
    print(f"Saved {len(feedback_data)} feedback items to {filename}")


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0
