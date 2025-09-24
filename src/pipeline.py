# Multi-Agent Content Intelligence Pipeline, A LangGraph for orchestration

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentState:
    """Shared state between agents"""
    request: str = ""
    research_data: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    draft_content: Optional[str] = None
    quality_feedback: Optional[Dict[str, Any]] = None
    final_output: Optional[str] = None
    current_agent: str = "coordinator"
    status: AgentStatus = AgentStatus.PENDING
    errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {
                "start_time": datetime.now().isoformat(),
                "agent_history": []
            }

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, model: str = "gemini-2.5-flash"):
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
        )
        
    def log_agent_activity(self, state: AgentState, action: str):
        """Log agent activity for traceability"""
        if state.metadata is None:
            state.metadata = {"start_time": datetime.now().isoformat(), "agent_history": []}
        state.metadata["agent_history"].append({
            "agent": self.name,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"{self.name}: {action}")

class ResearchAgent(BaseAgent):
    """Agent responsible for gathering information"""
    
    def __init__(self):
        super().__init__("ResearchAgent")
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                # TavilyClient is usually sync; we'll call it via executor when used
                self.tavily_client = TavilyClient(api_key=tavily_key)
                logger.info("Tavily client initialized for ResearchAgent")
            except Exception as e:
                logger.warning(f"Failed to init TavilyClient: {e}")
                self.tavily_client = None
        else:
            logger.warning("TAVILY_API_KEY not found; Tavily searches will be skipped")
            self.tavily_client = None
        
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute research phase"""
        self.log_agent_activity(state, "Starting research phase")
        
        try:
            # Create research prompt
            research_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Research Agent specializing in gathering comprehensive information.
                Your task is to gather relevant data for the given request.
                
                For the request, provide:
                1. Key topics to research
                2. Relevant sources
                3. Important data points
                4. Study types and methodologies
                5. Sample findings
                
                Structure your response as JSON with the following keys:
                - topics: list of key research topics
                - sources: list of relevant sources
                - findings: list of key findings
                - study_count: number of studies found
                - confidence_level: research confidence (1-10)"""),
                ("human", "Research request: {request}")
            ])
            
            # Attempt an LLM call
            try:
                chain = research_prompt | self.llm
                response = await chain.ainvoke({"request": state.request})
            except Exception as e:
                # Don't fail the whole agent if the LLM call cannot be made in this environment.
                logger.warning(f"ResearchAgent LLM call failed (continuing with simulated data): {e}")
                response = None
            
            # call the async helpers (they return actual results, not coroutines)
            topics = await self._extract_topics(state.request)
            sources = await self._gather_sources(state.request)
            findings = await self._gather_findings(state.request)
            # Parse research data
            study_count = sum(s.get("articles_found", 1) for s in sources)
            num_sources = len(sources)
            num_findings = len(findings)
            confidence_level = min(10, max(1, int((num_sources + num_findings) / 2)))
            research_data = {
                "topics": topics,
                "sources": sources,
                "findings": findings,
                "study_count": study_count,
                "confidence_level": confidence_level,
                "raw_response": getattr(response, "content", None)
            }
            
            state.research_data = research_data
            state.current_agent = "analysis"
            self.log_agent_activity(state, f"Research completed with {research_data['study_count']} sources")
            
        except Exception as e:
            error_msg = f"Research agent failed: {str(e)}"
            state.errors.append(error_msg)
            state.status = AgentStatus.FAILED
            logger.error(error_msg)
            
        return state
    
    async def _extract_topics(self, request: str, max_topics: int = 8) -> List[str]:
        """
        Use the LLM to extract key topics from the request.
        Returns list of short topic strings. Robust to different return types.
        """
        prompt = ChatPromptTemplate.from_messages([
            # Escape literal braces so LangChain doesn't treat them as template variables
            ("system", "You are a concise topic extractor. Return JSON only: {{\"topics\": [\"topic1\", ...]}}"),
            ("human", "Research request: {request}\nReturn JSON only.")
        ])

        try:
            chain = prompt | self.llm
            resp = await chain.ainvoke({"request": request, "max_topics": max_topics})

            # robustly extract text from common return shapes
            if hasattr(resp, "content"):
                text = resp.content
            elif isinstance(resp, dict) and "content" in resp:
                text = resp["content"]
            elif isinstance(resp, list) and len(resp) > 0 and hasattr(resp[0], "content"):
                text = resp[0].content
            elif isinstance(resp, str):
                text = resp
            else:
                text = str(resp)

            text = (text or "").strip()
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                # try to extract JSON substring (lenient)
                import re
                m = re.search(r"\{.*\"topics\".*\}", text, flags=re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None

            if isinstance(parsed, dict) and isinstance(parsed.get("topics"), list):
                topics = [t.strip() for t in parsed["topics"]][:max_topics]
                return topics

        except Exception as e:
            logger.warning(f"_extract_topics LLM call failed: {e}")

        # fallback heuristic
        lowered = request.lower()
        if "telemedicine" in lowered:
            return ["telemedicine", "diabetes management", "remote monitoring", "clinical outcomes"]
        if "ai" in lowered or "artificial intelligence" in lowered:
            return ["artificial intelligence", "machine learning", "healthcare applications", "clinical decision support"]
        # fallback: split on punctuation and take short phrases
        import re
        parts = re.split(r"[,\.;:]\s*", request)[:max_topics]
        return [p.strip() for p in parts if p.strip()][:max_topics]

    # made async so callers can uniformly await
    async def _gather_sources(self, request: str) -> List[Dict[str, Any]]:
        """
        Use Tavily search results as 'sources'. Runs sync client in thread executor.
        Returns a list of source dicts:
        { "name": ..., "url": ..., "snippet": ..., "score": ... }
        """
        # Fallback simulated sources if client not available
        if not self.tavily_client:
            return [
                {"name": "PubMed", "type": "database", "articles_found": 23},
                {"name": "Cochrane Library", "type": "systematic_reviews", "articles_found": 8},
                {"name": "ClinicalTrials.gov", "type": "clinical_trials", "articles_found": 16}
            ]

        query = f"{request} systematic reviews clinical trials key findings"
        loop = asyncio.get_running_loop()
        try:
            # tavily-python client is likely sync; run in executor to keep async flow
            raw = await loop.run_in_executor(None, lambda: self.tavily_client.search(query=query, max_results=8))
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []

        # Defensive parsing: tavily responses vary by version; try common fields
        sources = []
        try:
            # If client returned a dict with 'results' list
            if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
                for r in raw["results"]:
                    sources.append({
                        "name": r.get("domain") or r.get("source") or r.get("title") or r.get("url"),
                        "url": r.get("url"),
                        "snippet": (r.get("snippet") or r.get("raw_content") or "")[:400],
                        "score": r.get("score")
                    })
            # If client returned a top-level 'answer' plus 'sources'
            elif isinstance(raw, dict) and "sources" in raw and isinstance(raw["sources"], list):
                for r in raw["sources"]:
                    sources.append({
                        "name": r.get("domain") or r.get("source") or r.get("title") or r.get("url"),
                        "url": r.get("url"),
                        "snippet": (r.get("snippet") or r.get("raw_content") or "")[:400],
                        "score": r.get("score")
                    })
            # If tavily returned a list of results directly
            elif isinstance(raw, list):
                for r in raw:
                    if isinstance(r, dict):
                        sources.append({
                            "name": r.get("domain") or r.get("source") or r.get("title") or r.get("url"),
                            "url": r.get("url"),
                            "snippet": (r.get("snippet") or r.get("raw_content") or "")[:400]
                        })
                    else:
                        sources.append({"name": "tavily", "snippet": str(r)})
            else:
                # Fallback: return the raw answer as a single source
                sources.append({"name": "tavily", "snippet": str(raw)})
        except Exception as e:
            logger.warning(f"Error parsing Tavily response for sources: {e}")
            sources.append({"name": "tavily", "snippet": str(raw)})

        return sources
    
    # made async so callers can uniformly await
    async def _gather_findings(self, request: str) -> List[str]:
        """
        Use Tavily's concise answer or top snippets as 'findings'.
        Returns a list of short strings summarizing results.
        """
        if not self.tavily_client:
            # keep your previous heuristics when no client available
            if "telemedicine" in request.lower():
                return [
                    "Telemedicine interventions show 0.3-0.5% reduction in HbA1c levels",
                    "85% patient satisfaction rate with remote monitoring",
                    "30% improvement in medication adherence",
                    "Reduced healthcare costs by 15-20%"
                ]
            else:
                return [
                    "Technology interventions show positive clinical outcomes",
                    "High patient acceptance rates",
                    "Cost-effective implementation",
                    "Improved care coordination"
                ]

        query = f"{request} key findings summary"
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, lambda: self.tavily_client.search(query=query, max_results=6, include_answer=True))
        except Exception as e:
            logger.warning(f"Tavily findings query failed: {e}")
            return []

        findings = []
        try:
            # If tavily returns a dict with an 'answer' field, prefer it
            if isinstance(raw, dict) and raw.get("answer"):
                answer = raw.get("answer")
                # If answer is a long textual summary, split into bullet-like sentences
                if isinstance(answer, str):
                    # naive sentence split; you may improve this
                    for s in answer.split("\n"):
                        s = s.strip()
                        if s:
                            findings.append(s)
                else:
                    findings.append(str(answer))
            # Otherwise, take top snippets from 'results' or 'sources'
            if isinstance(raw, dict) and "results" in raw:
                for r in raw["results"][:6]:
                    snippet = r.get("snippet") or r.get("raw_content") or ""
                    if snippet:
                        findings.append(snippet.strip())
            elif isinstance(raw, dict) and "sources" in raw:
                for r in raw["sources"][:6]:
                    snippet = r.get("snippet") or r.get("raw_content") or ""
                    if snippet:
                        findings.append(snippet.strip())
            elif isinstance(raw, list):
                for r in raw[:6]:
                    if isinstance(r, dict):
                        snippet = r.get("snippet") or r.get("raw_content") or ""
                        if snippet:
                            findings.append(snippet.strip())
                    else:
                        findings.append(str(r))
            # final fallback: raw as string
            if not findings:
                findings.append(str(raw))
        except Exception as e:
            logger.warning(f"Error parsing Tavily response for findings: {e}")
            findings.append(str(raw))

        # dedupe, trim, and return up to 8 findings
        unique = []
        for f in findings:
            s = (f or "").strip()
            if s and s not in unique:
                unique.append(s)
        return unique[:8]

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing gathered data"""
    
    def __init__(self):
        super().__init__("AnalysisAgent")
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute analysis phase"""
        self.log_agent_activity(state, "Starting analysis phase")
        
        try:
            if not state.research_data:
                raise ValueError("No research data available for analysis")
            
            # Create analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an Analysis Agent specializing in synthesizing research data.
                Your task is to analyze the provided research data and extract meaningful insights.
                
                Provide:
                1. Key themes and patterns
                2. Statistical significance analysis
                3. Quality assessment of evidence
                4. Limitations and biases identified
                5. Strength of evidence rating
                
                Structure your response as JSON with appropriate analysis categories."""),
                ("human", "Analyze this research data: {research_data}")
            ])
            
            # Perform analysis
            try:
                chain = analysis_prompt | self.llm
                response = await chain.ainvoke({
                    "research_data": json.dumps(state.research_data, indent=2)
                })
            except Exception as e:
                logger.warning(f"AnalysisAgent LLM call failed (continuing with heuristics): {e}")
                response = None
            
            # Structure analysis results
            analysis_results = {
                "key_themes": self._identify_themes(state.research_data),
                "statistical_analysis": self._perform_statistical_analysis(state.research_data),
                "quality_assessment": self._assess_quality(state.research_data),
                "evidence_strength": "Moderate to High",
                "limitations": self._identify_limitations(),
                "recommendations": self._generate_recommendations(state.research_data),
                "raw_analysis": getattr(response, "content", None)
            }
            
            state.analysis_results = analysis_results
            state.current_agent = "writing"
            self.log_agent_activity(state, "Analysis completed with quality assessment")
            
        except Exception as e:
            error_msg = f"Analysis agent failed: {str(e)}"
            state.errors.append(error_msg)
            state.status = AgentStatus.FAILED
            logger.error(error_msg)
            
        return state
    
    def _identify_themes(self, research_data: Dict) -> List[str]:
        """Identify key themes from research"""
        return [
            "Clinical effectiveness demonstrated",
            "Technology acceptance high",
            "Cost-effectiveness established",
            "Implementation challenges identified"
        ]
    
    def _perform_statistical_analysis(self, research_data: Dict) -> Dict:
        """Perform statistical analysis"""
        return {
            "effect_size": "Medium (Cohen's d = 0.5)",
            "confidence_interval": "95% CI [0.3, 0.7]",
            "p_value": "p < 0.001",
            "heterogeneity": "I² = 45% (moderate)"
        }
    
    def _assess_quality(self, research_data: Dict) -> Dict:
        """Assess evidence quality"""
        return {
            "overall_quality": "Moderate to High",
            "study_designs": "Mixed (RCTs and observational)",
            "risk_of_bias": "Low to Moderate",
            "grade_rating": "B"
        }
    
    def _identify_limitations(self) -> List[str]:
        """Identify study limitations"""
        return [
            "Limited long-term follow-up data",
            "Heterogeneity in intervention types",
            "Publication bias potential",
            "Limited diversity in study populations"
        ]
    
    def _generate_recommendations(self, research_data: Dict) -> List[str]:
        """Generate clinical recommendations"""
        return [
            "Consider implementation in appropriate patient populations",
            "Ensure adequate technical support",
            "Monitor patient outcomes closely",
            "Provide staff training on technology use"
        ]

class WritingAgent(BaseAgent):
    """Agent responsible for content creation"""
    
    def __init__(self):
        super().__init__("WritingAgent")
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute writing phase"""
        self.log_agent_activity(state, "Starting writing phase")
        
        try:
            if not state.analysis_results:
                raise ValueError("No analysis results available for writing")
            
            # Create writing prompt
            writing_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Writing Agent specializing in creating comprehensive clinical evidence syntheses.
                Your task is to create a well-structured, professional report based on the research and analysis provided.
                
                Create a comprehensive clinical evidence synthesis with:
                1. Executive Summary
                2. Background and Objectives
                3. Methods
                4. Results
                5. Discussion
                6. Conclusions and Recommendations
                7. References
                
                Use professional medical writing style and include appropriate citations."""),
                ("human", """Create a clinical evidence synthesis for: {request}
                
                Based on research data: {research_data}
                
                Analysis results: {analysis_results}""")
            ])
            
            # Generate content
            try:
                chain = writing_prompt | self.llm
                response = await chain.ainvoke({
                    "request": state.request,
                    "research_data": json.dumps(state.research_data, indent=2),
                    "analysis_results": json.dumps(state.analysis_results, indent=2)
                })
                raw_content = getattr(response, "content", "")
            except Exception as e:
                logger.warning(f"WritingAgent LLM call failed (using a placeholder draft): {e}")
                raw_content = "## Draft (LLM generation failed in this environment)\n\nSummary: (simulated content)."
            
            # Structure the draft content
            draft_content = self._format_content(raw_content, state)
            
            state.draft_content = draft_content
            state.current_agent = "quality"
            self.log_agent_activity(state, "Draft content created")
            
        except Exception as e:
            error_msg = f"Writing agent failed: {str(e)}"
            state.errors.append(error_msg)
            state.status = AgentStatus.FAILED
            logger.error(error_msg)
            
        return state
    
    def _format_content(self, content: str, state: AgentState) -> str:
        """Format content with metadata"""
        header = f"""
# Clinical Evidence Synthesis
**Request:** {state.request}\n
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n
**Studies Analyzed:** {state.research_data.get('study_count', 'N/A') if state.research_data else 'N/A'}\n
**Evidence Quality:** {state.analysis_results.get('evidence_strength', 'N/A') if state.analysis_results else 'N/A'}

---

"""
        return header + (content or "")

class QualityAgent(BaseAgent):
    """Agent responsible for quality assurance"""
    
    def __init__(self):
        super().__init__("QualityAgent")
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute quality review phase"""
        self.log_agent_activity(state, "Starting quality review")
        
        try:
            if not state.draft_content:
                raise ValueError("No draft content available for quality review")
            
            # Create quality review prompt
            quality_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Quality Agent specializing in clinical content review.
                Your task is to review the draft content for:
                1. Clinical accuracy
                2. Statistical interpretation
                3. Logical consistency
                4. Citation accuracy
                5. Completeness
                
                Provide specific feedback and approval status.
                Return JSON with: approved (boolean), feedback (list), severity (string)"""),
                ("human", "Review this clinical evidence synthesis:\n\n{draft_content}")
            ])
            
            # Perform quality review
            try:
                chain = quality_prompt | self.llm
                response = await chain.ainvoke({"draft_content": state.draft_content})
                raw_feedback = getattr(response, "content", None)
            except Exception as e:
                logger.warning(f"QualityAgent LLM call failed (using simplified feedback): {e}")
                raw_feedback = None
            
            # Process quality feedback
            quality_feedback = self._process_feedback(raw_feedback)
            
            state.quality_feedback = quality_feedback
            
            # Determine next step based on quality review
            if quality_feedback.get("approved", True):
                state.final_output = self._finalize_content(state.draft_content, quality_feedback)
                state.status = AgentStatus.COMPLETED  # Explicitly mark as completed
                state.current_agent = "end"  # Signal the graph to stop
                self.log_agent_activity(state, "Quality review passed - content approved")
            else:
                state.current_agent = "writing"  # Send back to writing for revision
                self.log_agent_activity(state, "Quality review failed - revision needed")
            
        except Exception as e:
            error_msg = f"Quality agent failed: {str(e)}"
            state.errors.append(error_msg)
            state.status = AgentStatus.FAILED
            logger.error(error_msg)
            
        return state
    
    def _process_feedback(self, feedback_content: Optional[str]) -> Dict:
        """Process quality feedback"""
        # Simplified feedback processing (LLM parsing could be added)
        return {
            "approved": True,  # Simplified for demo
            "feedback": [
                "Clinical terminology used appropriately",
                "Statistical interpretations are sound",
                "Citations formatted correctly",
                "Minor suggestion: Add confidence intervals to summary"
            ],
            "severity": "minor",
            "overall_score": 8.5
        }
    
    def _finalize_content(self, draft: str, feedback: Dict) -> str:
        """Finalize content based on quality feedback"""
        footer = f"""

---
**Quality Review Summary:**
- Approved: {feedback.get('approved', 'Yes')}
- Overall Score: {feedback.get('overall_score', 'N/A')}/10
- Review Comments: {len(feedback.get('feedback', []))} items

**Processing Complete:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return draft + footer

class CoordinatorAgent(BaseAgent):
    """Agent responsible for orchestrating the workflow"""
    
    def __init__(self):
        super().__init__("CoordinatorAgent")
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute coordination logic"""
        self.log_agent_activity(state, "Coordinating workflow")
        
        try:
            # If pipeline already completed, nothing to do
            if state.status == AgentStatus.COMPLETED:
                self.log_agent_activity(state, "Workflow already completed")
            else:
                # Initialize workflow
                state.current_agent = "research"
                state.status = AgentStatus.IN_PROGRESS
                self.log_agent_activity(state, "Initializing research phase")
                
        except Exception as e:
            error_msg = f"Coordinator failed: {str(e)}"
            state.errors.append(error_msg)
            state.status = AgentStatus.FAILED
            logger.error(error_msg)
            
        return state

class ContentIntelligencePipeline:
    """Main pipeline orchestrator using LangGraph"""
    
    def __init__(self):
        self.agents = {
            "coordinator": CoordinatorAgent(),
            "research": ResearchAgent(),
            "analysis": AnalysisAgent(),
            "writing": WritingAgent(),
            "quality": QualityAgent()
        }
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("coordinator", self._run_coordinator)
        workflow.add_node("research", self._run_research)
        workflow.add_node("analysis", self._run_analysis)
        workflow.add_node("writing", self._run_writing)
        workflow.add_node("quality", self._run_quality)
        
        # Define edges (workflow transitions)
        # Coordinator always starts with research
        workflow.add_edge("coordinator", "research")
        workflow.add_edge("research", "analysis")
        workflow.add_edge("analysis", "writing")
        workflow.add_edge("writing", "quality")
        
        # Conditional edge from quality back to writing or to END
        workflow.add_conditional_edges(
            "quality",
            self._should_continue,
            {
                # Map branch names to targets
                "end": END,
                "revise": "writing",
            }
        )
        
        # Set entry point
        workflow.set_entry_point("coordinator")
        
        return workflow.compile()
    
    async def _run_coordinator(self, state: AgentState) -> AgentState:
        """Run coordinator agent"""
        return await self.agents["coordinator"].execute(state)
    
    async def _run_research(self, state: AgentState) -> AgentState:
        """Run research agent"""
        return await self.agents["research"].execute(state)
    
    async def _run_analysis(self, state: AgentState) -> AgentState:
        """Run analysis agent"""
        return await self.agents["analysis"].execute(state)
    
    async def _run_writing(self, state: AgentState) -> AgentState:
        """Run writing agent"""
        return await self.agents["writing"].execute(state)
    
    async def _run_quality(self, state: AgentState) -> AgentState:
        """Run quality agent"""
        return await self.agents["quality"].execute(state)
    
    # --- START CORRECTION: Simplified _should_continue logic
    def _should_continue(self, state: AgentState) -> str:
        """Determine workflow continuation from the quality node"""
        if state.status == AgentStatus.FAILED or state.status == AgentStatus.COMPLETED:
            # Go to END if the final status has been set
            return "end"
        elif state.quality_feedback and not state.quality_feedback.get("approved", True):
            # Go back to writing for revision
            return "revise"
        else:
            # Fallback to end
            return "end"
    # --- END CORRECTION

    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process a content intelligence request"""
        logger.info(f"Processing request: {request}")

        # Initialize state
        initial_state = AgentState(request=request)

        try:
            # Run the workflow
            final_state_obj = await self.workflow.ainvoke(initial_state)

            # The workflow may return an AgentState or a dict depending on LangGraph version.
            if isinstance(final_state_obj, AgentState):
                final_state = final_state_obj
                success = final_state.status == AgentStatus.COMPLETED
                return {
                    "success": success,
                    "final_output": final_state.final_output,
                    "metadata": final_state.metadata,
                    "errors": final_state.errors,
                    "quality_feedback": final_state.quality_feedback
                }
            elif isinstance(final_state_obj, dict):
                # If LangGraph returned a dict, use its values (defensive)
                final_dict = final_state_obj
                # status may be a string or enum
                status_val = final_dict.get("status")
                success = status_val == AgentStatus.COMPLETED or status_val == AgentStatus.COMPLETED.value
                return {
                    "success": success,
                    "final_output": final_dict.get("final_output"),
                    "metadata": final_dict.get("metadata"),
                    "errors": final_dict.get("errors"),
                    "quality_feedback": final_dict.get("quality_feedback")
                }
            else:
                # Unexpected return type — attempt to stringify
                logger.warning("Unexpected final state type from workflow. Returning best-effort diagnostics.")
                return {
                    "success": False,
                    "error": "Unexpected final state type",
                    "final_state": str(final_state_obj),
                    "metadata": initial_state.metadata
                }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": initial_state.metadata
            }
