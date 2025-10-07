import os
import gradio as gr
import requests
import inspect
import pandas as pd
from agent import build_graph
from langchain_core.messages import HumanMessage

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class GAIAAgent:
    def __init__(self):
        print("GAIAAgent initialized - building LangGraph agent...")
        self.graph = build_graph(provider="vertexai")
        print("LangGraph agent built successfully.")
        
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        try:
            # Invoke the graph with the question
            result = self.graph.invoke({"messages": [HumanMessage(content=question)]})
            
            # Extract the final answer from the last message
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1].content
                # Look for FINAL ANSWER in the response
                if "FINAL ANSWER:" in last_message:
                    answer = last_message.split("FINAL ANSWER:")[-1].strip()
                else:
                    answer = last_message
                print(f"Agent returning answer: {answer[:100]}...")
                return answer
            else:
                return "No response generated"
        except Exception as e:
            print(f"Error running agent: {e}")
            return f"Error: {str(e)}"

def run_and_submit_all():
    """
    Fetches all questions, runs the GAIAAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    # For local testing, use a default username
    username = os.getenv("HF_USERNAME", "local_user")
    print(f"Running as: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = GAIAAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    if space_id:
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    else:
        agent_code = "local_development"
    print(f"Agent code location: {agent_code}")

    # 2. Fetch Questions and Download Associated Files
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
        
        # Download files for questions that have them
        files_url = f"{api_url}/files"
        for item in questions_data:
            task_id = item.get("task_id")
            file_name = item.get("file_name", "")
            
            if file_name:  # If there's a file associated with this question
                print(f"Downloading file for task {task_id}: {file_name}")
                try:
                    file_response = requests.get(f"{files_url}/{task_id}", timeout=30)
                    file_response.raise_for_status()
                    
                    # Determine file extension from content type or file_name
                    content_type = file_response.headers.get('content-type', '')
                    if not file_name:
                        if 'image' in content_type:
                            file_name = f"{task_id}.png"
                        elif 'audio' in content_type:
                            file_name = f"{task_id}.mp3"
                        elif 'excel' in content_type or 'spreadsheet' in content_type:
                            file_name = f"{task_id}.xlsx"
                        elif 'python' in content_type or 'text' in content_type:
                            file_name = f"{task_id}.py"
                        else:
                            file_name = f"{task_id}.bin"
                    
                    # Save the file
                    with open(file_name, 'wb') as f:
                        f.write(file_response.content)
                    
                    # Add file path to the item
                    item['file_path'] = file_name
                    print(f"  Downloaded: {file_name} ({len(file_response.content)} bytes)")
                    
                except requests.exceptions.RequestException as e:
                    print(f"  Error downloading file for {task_id}: {e}")
                    item['file_path'] = None
                    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_path = item.get("file_path", None)
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        # Add file path information to the question if a file exists
        if file_path:
            enhanced_question = f"{question_text}\n\nFile available at: {file_path}"
        else:
            enhanced_question = question_text
            
        try:
            print(f"Processing task {task_id}...")
            submitted_answer = agent(enhanced_question)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
            print(f"  Answer: {submitted_answer[:100]}..." if len(submitted_answer) > 100 else f"  Answer: {submitted_answer}")
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. Your agent is configured to use Google VertexAI Gemini model
        2. Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        3. Note: This can take some time as the agent processes all questions.

        ---
        **Setup:**
        - Model: Gemini 2.5 Pro (VertexAI)
        - Tools: Wikipedia, Web Search (Tavily), ArXiv, Math operations
        - Vector Store: ChromaDB (for similar question retrieval)
        """
    )

    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        inputs=[],
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)