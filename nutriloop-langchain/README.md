# NutriLoop LangChain

NutriLoop LangChain is a personalized nutrition application that leverages the power of LangChain and LangGraph to create tailored meal plans and assess the safety of dietary goals. This project integrates various components, including agents for nutrition and safety, tools for communication and scheduling, and validation mechanisms to ensure user preferences and constraints are respected.

## Features

- **Personalized Meal Planning**: Generate weekly meal plans based on user preferences, allergies, and budget constraints.
- **Safety Assessment**: Evaluate the safety of dietary goals and provide feedback to ensure health standards are met.
- **Integration with LangChain and LangGraph**: Utilize advanced language model capabilities for decision-making and workflow management.
- **Email and Calendar Tools**: Seamlessly communicate with users and schedule appointments.

## Project Structure

```
nutriloop-langchain
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── orchestrator.py
│   ├── llm
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── chain_setup.py
│   ├── agents
│   │   ├── __init__.py
│   │   ├── nutrition_agent.py
│   │   └── safety_agent.py
│   ├── flows
│   │   ├── __init__.py
│   │   └── langgraph_flow.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── mail.py
│   │   ├── calendar.py
│   │   └── grocery.py
│   ├── validators.py
│   └── utils.py
├── tests
│   ├── test_validators.py
│   └── test_orchestrator.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nutriloop-langchain.git
   cd nutriloop-langchain
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by copying `.env.example` to `.env` and updating the values as needed.

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will start the workflow, prompting for user input to generate a personalized meal plan.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.