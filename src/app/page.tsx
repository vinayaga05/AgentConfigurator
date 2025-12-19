"use client";

import { useCoAgent, useCopilotAction } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useState } from "react";

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/guides/frontend-actions
  useCopilotAction({
    name: "setThemeColor",
    parameters: [{
      name: "themeColor",
      description: "The theme color to set. Make sure to pick nice colors.",
      required: true, 
    }],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties}>
      <YourMainContent themeColor={themeColor} />
      <CopilotSidebar
        clickOutsideToClose={false}
        defaultOpen={true}
        labels={{
          title: "Popup Assistant",
          initial: "👋 Hi, there! You're chatting with an agent configuration assistant. This agent helps you create and configure AI agents.\n\nFor example you can try:\n- **Create Agent**: \"Create a customer support agent\"\n- **Frontend Tools**: \"Set the theme to orange\"\n- **Generative UI**: \"Get the weather in SF\"\n\nAs you interact with the agent, you'll see the UI update in real-time to show the agent **instruction**, **tool calls**, and **progress**."
        }}
      />
    </main>
  );
}

// State of the agent, make sure this aligns with your agent's state.
type AgentInstruction = {
  role?: string;
  responsibility?: string;
  process_title?: string;
  process_steps?: string[];
  tool_usage_title?: string;
  tool_steps?: string[];
}

type RecommendedTool = {
  tool_id?: string;
  tool_name?: string;
  app_name?: string;
  description?: string;
  category?: string;
  [key: string]: any; // Allow additional properties
}

type AgentState = {
  instruction?: AgentInstruction;
  recommended_tools?: RecommendedTool[];
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // 🪁 Shared State: https://docs.copilotkit.ai/coagents/shared-state
  const { state, setState } = useCoAgent<AgentState>({
    name: "sample_agent",
    initialState: {
      instruction: undefined,
      recommended_tools: [],
    },
  })

  // Frontend action to update agent instruction
  useCopilotAction({
    name: "updateAgentInstruction",
    description: "Update the agent instruction with role, responsibility, and process steps.",
    parameters: [{
      name: "instruction",
      description: "The agent instruction object with role, responsibility, process_steps, etc.",
      type: "object",
      required: true,
    }],
    handler: ({ instruction }) => {
      setState({
        ...state,
        instruction: instruction,
      });
    },
  });

  // Frontend action to update recommended tools
  useCopilotAction({
    name: "updateRecommendedTools",
    description: "Update the list of recommended tools for the agent.",
    parameters: [{
      name: "recommended_tools",
      description: "Array of recommended tool objects with tool_id, tool_name, app_name, description, etc.",
      type: "object[]",
      required: true,
    }],
    handler: ({ recommended_tools }) => {
      // Ensure recommended_tools is an array
      const toolsArray = Array.isArray(recommended_tools) ? recommended_tools : [];
      setState({
        ...state,
        recommended_tools: toolsArray as RecommendedTool[],
      });
    },
  });

  //🪁 Generative UI: https://docs.copilotkit.ai/coagents/generative-ui
  useCopilotAction({
    name: "get_weather",
    description: "Get the weather for a given location.",
    available: "disabled",
    parameters: [
      { name: "location", type: "string", required: true },
    ],
    render: ({ args }) => {
      return <WeatherCard location={args.location} themeColor={themeColor} />
    },
  });

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="h-screen w-screen flex justify-center items-start py-8 transition-colors duration-300 overflow-y-auto"
    >
      <div className="max-w-6xl w-full px-4 flex flex-col lg:flex-row gap-6">
        {/* Left Column - Agent Instruction */}
        <div className="bg-white/20 backdrop-blur-md p-8 rounded-2xl shadow-xl flex-1 max-h-[90vh] overflow-y-auto">
          <h1 className="text-4xl font-bold text-white mb-2 text-center">Agent Instruction</h1>
          <p className="text-gray-200 text-center italic mb-6">Configure your AI agent with role, responsibilities, and process steps 🤖</p>
          <hr className="border-white/20 my-6" />
          
          {state.instruction ? (
            <div className="flex flex-col gap-6">
              {/* Role Section */}
              {state.instruction.role && (
                <div className="bg-white/15 p-5 rounded-xl text-white">
                  <h2 className="text-xl font-semibold mb-2 text-yellow-200">Role</h2>
                  <p className="text-white/90">{state.instruction.role}</p>
                </div>
              )}

              {/* Responsibility Section */}
              {state.instruction.responsibility && (
                <div className="bg-white/15 p-5 rounded-xl text-white">
                  <h2 className="text-xl font-semibold mb-2 text-yellow-200">Responsibility</h2>
                  <p className="text-white/90">{state.instruction.responsibility}</p>
                </div>
              )}

              {/* Process Steps Section */}
              {state.instruction.process_steps && state.instruction.process_steps.length > 0 && (
                <div className="bg-white/15 p-5 rounded-xl text-white">
                  <h2 className="text-xl font-semibold mb-3 text-yellow-200">
                    {state.instruction.process_title || "Main Process"}
                  </h2>
                  <ol className="list-decimal list-inside space-y-2 text-white/90">
                    {state.instruction.process_steps.map((step, index) => (
                      <li key={index} className="pl-2">{step}</li>
                    ))}
                  </ol>
                </div>
              )}

              {/* Tool Steps Section */}
              {state.instruction.tool_steps && state.instruction.tool_steps.length > 0 && (
                <div className="bg-white/15 p-5 rounded-xl text-white">
                  <h2 className="text-xl font-semibold mb-3 text-yellow-200">
                    {state.instruction.tool_usage_title || "Tool-Specific Operations"}
                  </h2>
                  <ol className="list-decimal list-inside space-y-2 text-white/90">
                    {state.instruction.tool_steps.map((step, index) => (
                      <li key={index} className="pl-2">{step}</li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-white/80 italic text-lg mb-4">
                No agent instruction yet.
              </p>
              <p className="text-white/60 text-sm">
                Ask the assistant to create an agent configuration!
              </p>
              <p className="text-white/60 text-sm mt-2">
                Try: "Create a customer support agent" or "I need an agent to send emails"
              </p>
            </div>
          )}
        </div>

        {/* Right Column - Recommended Tools */}
        <div className="bg-white/20 backdrop-blur-md p-8 rounded-2xl shadow-xl flex-1 max-h-[90vh] overflow-y-auto">
          <h1 className="text-4xl font-bold text-white mb-2 text-center">Recommended Tools</h1>
          <p className="text-gray-200 text-center italic mb-6">Tools selected for your agent configuration 🔧</p>
          <hr className="border-white/20 my-6" />
          
          {state.recommended_tools && state.recommended_tools.length > 0 ? (
            <div className="flex flex-col gap-4">
              {state.recommended_tools.map((tool, index) => (
                <div 
                  key={tool.tool_id || index} 
                  className="bg-white/15 p-5 rounded-xl text-white hover:bg-white/20 transition-all"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-yellow-200 mb-1">
                        {tool.tool_name || tool.tool_id || `Tool ${index + 1}`}
                      </h3>
                      {tool.app_name && (
                        <p className="text-sm text-white/70 mb-2">
                          App: <span className="font-medium">{tool.app_name}</span>
                        </p>
                      )}
                      {tool.description && (
                        <p className="text-white/90 text-sm leading-relaxed">
                          {tool.description}
                        </p>
                      )}
                      {tool.category && (
                        <span className="inline-block mt-2 px-3 py-1 bg-white/20 rounded-full text-xs text-white/80">
                          {tool.category}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-white/80 italic text-lg mb-4">
                No recommended tools yet.
              </p>
              <p className="text-white/60 text-sm">
                Tools will appear here once the agent configuration is complete.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Simple sun icon for the weather card
function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-14 h-14 text-yellow-200">
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeWidth="2" stroke="currentColor" />
    </svg>
  );
}

// Weather card component where the location and themeColor are based on what the agent
// sets via tool calls.
function WeatherCard({ location, themeColor }: { location?: string, themeColor: string }) {
  return (
    <div
    style={{ backgroundColor: themeColor }}
    className="rounded-xl shadow-xl mt-6 mb-4 max-w-md w-full"
  >
    <div className="bg-white/20 p-4 w-full">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white capitalize">{location}</h3>
          <p className="text-white">Current Weather</p>
        </div>
        <SunIcon />
      </div>
      
      <div className="mt-4 flex items-end justify-between">
        <div className="text-3xl font-bold text-white">70°</div>
        <div className="text-sm text-white">Clear skies</div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-white">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-white text-xs">Humidity</p>
            <p className="text-white font-medium">45%</p>
          </div>
          <div>
            <p className="text-white text-xs">Wind</p>
            <p className="text-white font-medium">5 mph</p>
          </div>
          <div>
            <p className="text-white text-xs">Feels Like</p>
            <p className="text-white font-medium">72°</p>
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}
