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

- ### License:
This project is licensed under the MIT License.
