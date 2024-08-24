# Querying-CSVs-and-Plot-Graphs-With-LLMs
This application leverages the power of LangChain and the Gemini 1.5 Flash model to facilitate the exploration and visualization of data contained within CSV files. By harnessing the capabilities of a Language Model (LLM), the app provides users with insightful data analysis and automatically generates relevant plots based on the underlying data.
Currently, the application is running on the free version of the Gemini API, which offers foundational capabilities for querying and analyzing data. However, upgrading to a more advanced model could significantly enhance the quality of the language model’s responses, providing deeper insights and more refined visualizations.

### Features:
1. CSV file upload and preprocessing
2. Data summary and statistical analysis
3. AI-generated insights and visualizations
4. Custom visualization creation
5. Interactive Q&A with the dataset

6. ### Requirements:
- Python 3.8+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- LangChain
- Google Generative AI

- 4. Set up your Google API key:
- Obtain a Google API key from the Google Cloud Console
- Set the environment variable:

    - **On macOS/Linux**:
        ```bash
        export GOOGLE_API_KEY=your_api_key_here
        ```

    - **On Windows**:
        ```bash
        set GOOGLE_API_KEY=your_api_key_here

"Setting Up Google API Key as an Environment Variable on Windows":

To securely store your Google API key as an environment variable on Windows, follow these steps:

Go to Windows Search:

Click on the Start menu or press Win + S.
Type Edit the system environment variables and select the corresponding option.
Open System Properties:

In the System Properties window, click on the Advanced tab.
Access Environment Variables:

Click on the Environment Variables... button at the bottom.
Create a New Environment Variable:

Under the User variables or System variables section (depending on whether you want the variable to be available only for your user or for all users), click on New....
Variable name: Enter the name for your API key (e.g., GOOGLE_API_KEY).
Variable value: Paste your API key in the value field.
Click OK to save the new environment variable.
Verify the Environment Variable in Command Prompt:

Open Command Prompt by pressing Win + R, typing cmd, and pressing Enter.
Type the following command to verify that the environment variable is set correctly:
sh
Copy code
echo %GOOGLE_API_KEY%
If set correctly, the command will display your API key.
By following these steps, you will have successfully added your Google API key as an environment variable on Windows.

- ### License:
This project is licensed under the MIT License.
