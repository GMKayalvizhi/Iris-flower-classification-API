# Slim base image: full Python image ships build tools and docs you don't need at runtime.
# "slim" keeps the final image small without losing pip/venv compatibility.
FROM python:3.11-slim

WORKDIR /app

# Copy ONLY requirements first. Docker caches layers — if requirements.txt hasn't
# changed, this layer (and the slow pip install) is reused on every rebuild,
# instead of reinstalling sklearn/pandas every time you change a .py file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app (this layer WILL invalidate on every code change,
# which is fine — it's fast).
COPY . .

EXPOSE 8000

# --host 0.0.0.0 is required here, not optional.
# 127.0.0.1 inside a container only accepts connections from INSIDE that same
# container's network namespace. Docker's port mapping (-p 8000:8000) forwards
# traffic from your host machine into the container as if it came from an
# external IP — so if uvicorn only listens on 127.0.0.1, that "external" traffic
# never reaches it and you get connection-refused even though the container is
# running fine. 0.0.0.0 means "listen on all interfaces inside this container,"
# which includes the one Docker's port-forwarding actually uses.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]