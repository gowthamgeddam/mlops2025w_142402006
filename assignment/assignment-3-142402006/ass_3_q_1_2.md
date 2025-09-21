# DS5619 Machine Learning System Operations

## Assignment 3

### 1. Identify and fix all the errors in the JSON data from a student management system
Done &dArr;
#### JSON without errors:

```json
{
    "students": [
        {
            "id": 101,
            "name": "Sarah Johnson",
            "courses": [
                "CS101",
                "MATH200",
                "ENG150"
            ],
            "gpa": 3.85,
            "active": true,
            "graduation_date": null
        },
        {
            "id": 102,
            "name": "Alex Chen",
            "courses": [
                "CS101",
                "CS102",
                "STAT101"
            ],
            "gpa": 3.92,
            "active": true,
            "advisor": null,
            "notes": "Excellent student with strong analytical skills"
        },
        {
            "id": "103",
            "name": "Maria Rodriguez",
            "courses": [],
            "gpa": 3.67,
            "active": false,
            "special_programs": [
                "honors",
                "research"
            ],
        }
    ],
    "last_updated": "2024-09-15T10:30:00Z",
    "total_students": 3
}
```

### 2. Your web application currently uses this TOML configuration file.

```toml
[server]
host = "0.0.0.0"
port = 8080
debug = false
max_connections = 1000

[database]
url = "postgresql://localhost:5432/myapp"
pool_size = 20
timeout = 30

[logging]
level = "info"
file = "/var/log/myapp.log"
max_size = "100MB"
rotate = true

[[feature_flags]]
name = "new_ui"
enabled = true
rollout_percentage = 25

[[feature_flags]]
name = "analytics"
enabled = false
rollout_percentage = 0

[cache]
redis_url = "redis://localhost:6379/0"
ttl = 3600
```


Analyze the configuration and answer the following questions:
#### a. How many feature flags are currently defined, and which ones are active?

There are 2. One named "new_ui" is enabled

#### b. What happens when the log file reaches 100MB?

The "rotate = true" setting indicates log rotation is enabled.

When the log file exceeds 100MB, it will be rotated — typically, this means:

The current log file is archived (e.g., renamed with a timestamp or version).

A new, empty log file is created to continue logging.

Old logs may be deleted or compressed depending on the logging framework used.

#### c. If you wanted to make the server accessible only from localhost, what should you change?

Change host = "0.0.0.0" to host = "127.0.0.1"

#### d. Calculate the total number of seconds that cached items will remain valid.
`ttl` (Time to Live) for `[cache]` is set to 3600. So 3600 seconds.

#### e. Explain the difference between the `[feature_flags]` and `[[feature_flags]]` syntax.
+ `[feature_flags]` defines a **table**, used for a single instance.
+ `[[feature_flags]]` defines an **array of tables**, used to specify multiple items with the same structure.

### 3. Design a ML pipeline using JSON and TOML with the following features:
+ a. Implement the model inference using Pytorch using pre-trained Resnet
34,50,101,152 layers. - 5 Marks
+ b. Specify the data source and model architecture using JSON - 3 marks
+ c. Define the model parameters such as learning rate, etc for each architecture
using TOML - 3 Marks
+ d. Integrate the subquestion (b) and ( c) leading to a pipeline - 3 Marks
+ e. Perform hyperparameter tuning (Grid search using JSON) by using learning rates
= [0.1, 0.01, 0.01], optimizers = [adam, sgd] and momentum = [0.5, 0.9] - 3 Marks

All code is available in [ml_pipline] folder