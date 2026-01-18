# pitchcraft-model

Placeholder README for the Pitchcraft model

## Setup

To get started with development, follow these steps:

**1. Clone the repository using Git:**

```bash
git clone <url>
```

You may need to generate a personal access token through GitHub to clone via HTTPS.

**2. Navigate to the root of the project and install the required dependencies:**

```bash
make install
```

If you add dependencies, add them to `requirements.txt` using `pipreqs` or `pip freeze`. I recommend `pipreqs` since it scans the repository for imports so no unnecessary dependencies are added accidentally:

```bash
pip install pipreqs
pipreqs --force .
```

**3. Configure `.env` file:**

Copy the `.env.sample` file into a new file called `.env` and configure the variables as needed. 

The variable `DB_RDS_CERT_PATH` is the path to the SSL certificate bundle for Amazon RDS. To connect to our database with SSL, we need a certificate bundle for Amazon RDS (read more [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.SSL.html#UsingWithRDS.SSL.CertificatesDownload)). `aws-rds-cert.pem` is using the bundle for `us-east-1`.

After these environement variables are set, you should be able to connect to AWS RDS database. Here is an example of how to read from the database with the `get_read_cursor` context manager:
```python
from src.data.db import get_read_cursor

query = """
    SELECT * FROM players WHERE id = %s
"""
params = []
params.append(434378)

with get_read_cursor() as cursor:
    cursor.execute(query, tuple(params))
    result = cursor.fetchone()
    print(result)
```
