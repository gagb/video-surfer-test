import asyncio

from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient


import cv2

def get_video_length(video_path: str) -> str:
    """
    Returns the length of the video in seconds.

    :param video_path: Path to the video file.
    :return: Duration of the video in seconds.

    You may need to install opencv-python package using `pip install opencv-python`.
    You may also need to install ffmpeg is correctly installed on your machine.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    return f"The video is {duration:.2f} seconds long."


async def main() -> None:
    """
    Main function to run the video agent.
    """

    SYSTEM_MESSAGE = "You are an agent that can answer questions about local video files. Reply with TERMINATE when done."

    # Define an agent
    video_agent = AssistantAgent(
        name="VideoSurferAgent",
        model_client=OpenAIChatCompletionClient(
          model="gpt-4o",
          # api_key = "your_openai_api_key"
          ),
        system_message=SYSTEM_MESSAGE,
        tools=[get_video_length]
        )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([video_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="Hi! Who are you? What is the length of video.mp4?")
    await Console(stream)

asyncio.run(main())