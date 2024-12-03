import asyncio

from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient


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
        )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([video_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="Hi! Who are you?")
    await Console(stream)

asyncio.run(main())