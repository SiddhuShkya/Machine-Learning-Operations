import time  # Used to add delays between retry attempts
import redis  # Redis client library for Python
from flask import Flask  # Flask framework to build a web application

app = Flask(__name__)  # Create a Flask application instance
cache = redis.Redis(host="redis", port=6379)  # Connect to Redis service


def get_hit_count():
    retries = 5  # Number of times to retry if Redis is unavailable
    while True:
        try:
            return cache.incr("hits")  # Increment and return the page hit count
        except redis.exceptions.ConnectionError as ex:
            if retries == 0:  # Stop retrying when attempts are exhausted
                raise ex
            retries -= 1  # Decrease retry count
            time.sleep(0.5)  # Wait before retrying


@app.route("/")  # Define the root URL endpoint
def hello():
    count = get_hit_count()  # Get the current hit count from Redis
    return (
        f"Hello Siddhartha Shakya! I have been seen {count} times."  # Return response
    )
