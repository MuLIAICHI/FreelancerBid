#!/usr/bin/env python3
"""
Freelancer.com Bidder Agent

This script extends the Freelancer.com Project Search AI Agent with bidding capabilities.
It can automatically analyze projects and submit bids based on AI analysis.

Requirements:
- freelancersdk package (pip install freelancersdk)
- Python-dotenv (pip install python-dotenv)
- pydantic-ai-slim (pip install 'pydantic-ai-slim[openai]')
- colorama (pip install colorama)

Configuration:
Create a .env file with the following variables:
FLN_URL=https://www.freelancer.com (or https://www.freelancer-sandbox.com for testing)
FLN_OAUTH_TOKEN=your_oauth_token_here
OPENAI_API_KEY=your_openai_api_key_here
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from colorama import init, Fore, Style
from dotenv import load_dotenv
from freelancersdk.session import Session
from freelancersdk.resources.projects import (
    search_projects, get_projects, get_project_by_id, get_jobs,
    create_search_projects_filter,
    create_get_projects_user_details_object,
    create_get_projects_project_details_object
)
from freelancersdk.resources.projects.projects import place_project_bid
from freelancersdk.resources.users import get_self_user_id, get_self
from freelancersdk.exceptions import (
    ProjectsNotFoundException, JobsNotFoundException
)
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Initialize colorama for cross-platform colored terminal output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API credentials from environment variables
FLN_URL = os.getenv("FLN_URL", "https://www.freelancer.com")
FLN_OAUTH_TOKEN = os.getenv("FLN_OAUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#----- Define Pydantic models for our AI agents -----#

class ProjectSummary(BaseModel):
    """A summary of a Freelancer.com project."""
    title: str = Field(description="The title of the project")
    description: str = Field(description="Brief summary of the project's key points")
    skills_required: str = Field(description="Skills required for the project")
    budget_range: str = Field(description="The budget range for the project")
    recommendation: str = Field(description="Recommendation on whether this is a good project to bid on")
    bid_confidence_score: int = Field(description="Score from 0-100 indicating confidence in bidding on this project")

class ProjectAnalysis(BaseModel):
    """Analysis of multiple projects from Freelancer.com."""
    project_summaries: List[ProjectSummary] = Field(description="Summaries of individual projects")
    common_skills: List[str] = Field(description="Common skills requested across projects")
    average_budget: str = Field(description="Average budget across all projects")
    market_insight: str = Field(description="Insight about the current market based on these projects")
    recommendation: str = Field(description="Overall recommendation based on the projects analyzed")
    top_projects_to_bid: List[int] = Field(description="Indices (1-based) of projects recommended for bidding", default=[])

class BidProposal(BaseModel):
    """A bid proposal for a Freelancer.com project."""
    project_id: int = Field(description="ID of the project to bid on")
    bid_amount: float = Field(description="Amount to bid in the project's currency")
    period: int = Field(description="Estimated completion time in days")
    description: str = Field(description="Bid proposal text")
    milestone_percentage: int = Field(description="Percentage of the bid amount for the first milestone (0-100)", ge=0, le=100)

#----- Dependencies for our agents -----#

@dataclass
class SearchAgentDeps:
    """Dependencies for the Freelancer search agent."""
    projects: List[Dict[str, Any]]
    job_categories: List[Dict[str, Any]]
    session: Session
    user_skills: List[str]
    user_profile: Dict[str, Any]

@dataclass
class BidderAgentDeps:
    """Dependencies for the Freelancer bidder agent."""
    project: Dict[str, Any]
    session: Session
    user_profile: Dict[str, Any]
    user_skills: List[str]
    bid_history: List[Dict[str, Any]] = None
    typical_bid_amount: float = None

#----- Create our AI agents -----#

# Project Analysis Agent
freelancer_agent = Agent[SearchAgentDeps, Union[ProjectSummary, ProjectAnalysis]](
    'openai:gpt-4o',
    deps_type=SearchAgentDeps,
    result_type=Union[ProjectSummary, ProjectAnalysis],
    system_prompt=(
        "You are an expert freelancer assistant who helps users analyze projects from Freelancer.com. "
        "Your goal is to help the user identify good projects to bid on and provide insights about the market. "
        "When analyzing projects, consider the following: clarity of project description, budget reasonableness, "
        "client's history if available, required skills matching with user skills, and project feasibility. "
        "For each project, provide a bid_confidence_score from 0-100, where 0 means 'do not bid' and 100 means "
        "'definitely bid'. Scores above 70 indicate recommended projects."
    )
)

# Bidder Agent
bidder_agent = Agent[BidderAgentDeps, BidProposal](
    'openai:gpt-4o',
    deps_type=BidderAgentDeps,
    result_type=BidProposal,
    system_prompt=(
        "You are an expert freelancer bidding assistant. Your job is to create compelling bid proposals "
        "for projects on Freelancer.com. Your proposals should be: professional, personalized to the project, "
        "highlight relevant skills and experience, address the client's specific needs, and propose a competitive "
        "but fair price. Make sure the proposal talks about specific deliverables and timelines."
    )
)

#----- Tools for the Project Analysis Agent -----#

@freelancer_agent.tool
async def get_project_details(ctx: RunContext[SearchAgentDeps], project_index: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific project by its index in the search results.
    
    Args:
        ctx: The context containing the search results.
        project_index: The index (1-based) of the project in the search results.
        
    Returns:
        A dictionary containing the detailed project information.
    """
    if not ctx.deps.projects:
        return {"error": "No projects available in search results"}
    
    if project_index < 1 or project_index > len(ctx.deps.projects):
        return {"error": f"Invalid project index. Please choose between 1 and {len(ctx.deps.projects)}"}
    
    try:
        project = ctx.deps.projects[project_index - 1]
        # Get additional details if available
        project_id = project.get('id')
        if project_id:
            # Create objects to get project and user details
            project_details = create_get_projects_project_details_object(
                full_description=True,
                jobs=True,
                attachments=True,
                location=True
            )
            user_details = create_get_projects_user_details_object(
                basic=True,
                reputation=True,
                display_info=True
            )
            
            # Get detailed project information
            detailed_project = get_project_by_id(
                ctx.deps.session,
                project_id,
                project_details,
                user_details
            )
            return detailed_project
        else:
            return project
    except Exception as e:
        return {"error": f"Error fetching project details: {str(e)}"}

@freelancer_agent.tool
async def get_job_category_name(ctx: RunContext[SearchAgentDeps], job_id: int) -> str:
    """
    Get the name of a job category by its ID.
    
    Args:
        ctx: The context containing the job categories.
        job_id: The ID of the job category.
        
    Returns:
        The name of the job category.
    """
    for job in ctx.deps.job_categories:
        if job.get('id') == job_id:
            return job.get('name', 'Unknown')
    return "Unknown job category"

@freelancer_agent.tool
async def check_skill_match(ctx: RunContext[SearchAgentDeps], project_skills: List[str]) -> Dict[str, Any]:
    """
    Check if the user's skills match the project's required skills.
    
    Args:
        ctx: The context containing the user's skills.
        project_skills: List of skills required by the project.
        
    Returns:
        A dictionary with match information.
    """
    user_skills = ctx.deps.user_skills
    if not user_skills:
        return {"match": False, "reason": "No user skills available"}
    
    # Normalize skills (lowercase)
    user_skills_norm = [s.lower().strip() for s in user_skills]
    project_skills_norm = [s.lower().strip() for s in project_skills]
    
    # Find matching skills
    matching_skills = [s for s in project_skills_norm if s in user_skills_norm]
    
    # Calculate match percentage
    match_percentage = len(matching_skills) / len(project_skills_norm) * 100 if project_skills_norm else 0
    
    return {
        "match": match_percentage >= 50,
        "match_percentage": match_percentage,
        "matching_skills": matching_skills,
        "missing_skills": [s for s in project_skills_norm if s not in user_skills_norm]
    }

#----- Tools for the Bidder Agent -----#

@bidder_agent.tool
async def get_typical_bid_amount(ctx: RunContext[BidderAgentDeps]) -> float:
    """
    Calculate a typical bid amount based on the user's bidding history and project budget.
    
    Args:
        ctx: The context containing the bid history and project.
        
    Returns:
        A suggested bid amount.
    """
    if ctx.deps.typical_bid_amount:
        return ctx.deps.typical_bid_amount
    
    project = ctx.deps.project
    budget = project.get('budget', {})
    min_budget = budget.get('minimum')
    max_budget = budget.get('maximum', min_budget)
    
    # If the project has a budget, use it as a basis
    if min_budget is not None and max_budget is not None:
        # Aim for the middle of the budget range by default
        typical_amount = (min_budget + max_budget) / 2
    else:
        # Default fallback based on project type/complexity
        typical_amount = 500  # Default amount
        
    # Adjust based on bid history if available
    if ctx.deps.bid_history:
        # Calculate average of past bids
        past_bids = [bid.get('amount', 0) for bid in ctx.deps.bid_history if bid.get('amount')]
        if past_bids:
            avg_past_bid = sum(past_bids) / len(past_bids)
            # Blend historical average with project budget
            typical_amount = (typical_amount + avg_past_bid) / 2
    
    return round(typical_amount, 2)

@bidder_agent.tool
async def estimate_completion_time(ctx: RunContext[BidderAgentDeps]) -> int:
    """
    Estimate completion time in days based on the project description.
    
    Args:
        ctx: The context containing the project.
        
    Returns:
        Estimated completion time in days.
    """
    project = ctx.deps.project
    description = project.get('description', '')
    
    # Simple keyword-based estimation
    if any(word in description.lower() for word in ['urgent', 'asap', 'immediately']):
        return 3  # Urgent projects
    elif any(word in description.lower() for word in ['complex', 'extensive', 'comprehensive']):
        return 14  # Complex projects
    else:
        return 7  # Default timeline
        
#----- Functions to create API session and check connection -----#

def create_session():
    """Create a session with the Freelancer API"""
    if not FLN_OAUTH_TOKEN:
        logger.error("ERROR: FLN_OAUTH_TOKEN not found in environment variables")
        sys.exit(1)
        
    logger.info(f"Connecting to {FLN_URL}...")
    return Session(oauth_token=FLN_OAUTH_TOKEN, url=FLN_URL)

def check_connection(session):
    """Check if we can connect to the API and get basic user info"""
    try:
        user_id = get_self_user_id(session)
        user_info = get_self(session)
        
        print(f"{Fore.GREEN}Successfully connected to Freelancer API!{Style.RESET_ALL}")
        print(f"User ID: {user_id}")
        print(f"Username: {user_info.get('username', 'Unknown')}")
        print(f"Display name: {user_info.get('display_name', 'Unknown')}")
        return user_info
    except Exception as e:
        logger.error(f"Error connecting to Freelancer API: {e}")
        return None

def get_user_skills(session):
    """Get the user's skills from their profile"""
    try:
        user_id = get_self_user_id(session)
        # This is a simplified approach; in a real implementation you would
        # need to use the appropriate endpoint to get the user's skills
        # Here we're just returning a list of example skills
        return ["Python", "JavaScript", "Web Development", "API Integration", "Data Analysis"]
    except Exception as e:
        logger.error(f"Error fetching user skills: {e}")
        return []

def list_job_categories(session):
    """List available job categories"""
    try:
        jobs = get_jobs(
            session,
            job_ids=[],
            seo_details=True,
            lang='en'
        )
        
        print("\nAvailable job categories:")
        print("-" * 80)
        print(f"{'ID':<8} {'Name':<40}")
        print("-" * 80)
        
        for job in sorted(jobs, key=lambda j: j['name']):
            print(f"{job['id']:<8} {job['name']:<40}")
        
        print("-" * 80)
        return jobs
            
    except JobsNotFoundException as e:
        logger.error(f"Error retrieving job categories: {e}")
        return None

# def find_projects(session, args):
#     """Find projects based on specified criteria"""
#     print(f"\n{Fore.CYAN}Searching for projects...{Style.RESET_ALL}")
    
#     # Get job IDs from args or use defaults
#     job_ids = args.job_ids if args.job_ids else []  
    
#     # Set up search parameters
#     search_filter = create_search_projects_filter(
#         project_types=["fixed"] if args.fixed_only else None,
#         min_avg_price=args.min_price,
#         max_avg_price=args.max_price,
#         sort_field=args.sort,
#         jobs=job_ids,
#         # include_contests=args.include_contests,
#         # Add this parameter to get only active projects
#         # frontend_statuses=["active", "open"],
#     )
    
#     # Get project details and user details objects
#     project_details = create_get_projects_project_details_object(
#         full_description=True,
#         jobs=True,
#         attachments=True,
#         location=True
#     )
    
#     user_details = create_get_projects_user_details_object(
#         basic=True,
#         reputation=True,
#         display_info=True
#     )
    
#     try:
        
#         # Search for projects
#         result = search_projects(
#             session,
#             query=args.query,
#             search_filter=search_filter,
#             project_details=project_details,
#             user_details=user_details,
#             limit=args.limit,
#             active_only=True,
#         )
        
#         if not result or not result['projects']:
#             print(f"{Fore.YELLOW}No projects found matching the criteria.{Style.RESET_ALL}")
#             return []
            
#         # Display all found projects
#         projects = result['projects']
#         print(f"{Fore.GREEN}Found {len(projects)} projects:{Style.RESET_ALL}")
#         print("-" * 80)
        
#         for i, project in enumerate(projects):
#             # Get budget info
#             budget = project.get('budget', {})
#             min_budget = budget.get('minimum', 'N/A')
#             max_budget = budget.get('maximum', min_budget)
            
#             if min_budget != 'N/A' and max_budget != 'N/A':
#                 budget_str = f"${min_budget}" if min_budget == max_budget else f"${min_budget}-${max_budget}"
#             else:
#                 budget_str = "Budget not specified"
            
#             # Get currency info
#             currency = project.get('currency', {})
#             currency_code = currency.get('code', 'USD')
            
#             # Get job categories
#             job_list = []
#             for job in project.get('jobs', []):
#                 job_list.append(job.get('name', ''))
#             job_str = ", ".join(job_list)
            
#             # Get description snippet
#             description = project.get('description', 'No description')
#             if len(description) > 100:
#                 description_snippet = description[:97] + "..."
#             else:
#                 description_snippet = description
            
#             # Format date
#             if 'time_submitted' in project:
#                 time_submitted = datetime.fromtimestamp(project['time_submitted'])
#                 date_str = time_submitted.strftime("%Y-%m-%d %H:%M")
#             else:
#                 date_str = "Unknown date"
            
#             # Get owner info
#             owner_id = project.get('owner_id')
#             owner_info = result.get('users', {}).get(str(owner_id), {})
#             owner_name = owner_info.get('display_name', 'Unknown')
            
#             # Display location if available
#             location = None
#             if project.get('local') and project.get('location'):
#                 loc = project.get('location', {})
#                 city = loc.get('city', '')
#                 country = loc.get('country', {}).get('name', '')
#                 if city and country:
#                     location = f"{city}, {country}"
#                 elif city:
#                     location = city
#                 elif country:
#                     location = country
            
#             # Build project URL
#             project_url = f"{session.url}/projects/{project.get('seo_url', project.get('id'))}"
            
#             # Print project info
#             print(f"{Fore.CYAN}[{i+1}] {Fore.WHITE}{project['title']}{Style.RESET_ALL} (ID: {project['id']})")
#             print(f"    Posted: {date_str} by {owner_name}")
#             print(f"    Budget: {budget_str} {currency_code}")
#             if location:
#                 print(f"    Location: {location}")
#             print(f"    Skills: {job_str}")
#             print(f"    Description: {description_snippet}")
#             print(f"    URL: {project_url}")
#             print("-" * 80)
            
#             # Save project details to file if requested
#             if args.save_details:
#                 save_project_details(session, project['id'], args.output_dir)
        
#         return projects
        
#     except ProjectsNotFoundException as e:
#         logger.error(f"Error finding projects: {e}")
#         return []

def find_projects(session, args):
    """Find projects based on specified criteria"""
    print(f"\n{Fore.CYAN}Searching for projects...{Style.RESET_ALL}")
    
    # Get job IDs from args or use defaults
    job_ids = args.job_ids if args.job_ids else []  
    
    # Set up search parameters
    search_filter = create_search_projects_filter(
        project_types=["fixed"] if args.fixed_only else None,
        min_avg_price=args.min_price,
        max_avg_price=args.max_price,
        sort_field=args.sort,
        jobs=job_ids,
        include_contests=args.include_contests
    )
    
    # Get project details and user details objects
    project_details = create_get_projects_project_details_object(
        full_description=True,
        jobs=True,
        attachments=True,
        location=True
    )
    
    user_details = create_get_projects_user_details_object(
        basic=True,
        reputation=True,
        display_info=True
    )
    
    try:
        # Search for projects
        result = search_projects(
            session,
            query=args.query,
            search_filter=search_filter,
            project_details=project_details,
            user_details=user_details,
            limit=args.limit,
            active_only=True  # Use this parameter if available
        )
        
        if not result or not result['projects']:
            print(f"{Fore.YELLOW}No projects found matching the criteria.{Style.RESET_ALL}")
            return []
            
        # Filter projects to only include those that are biddable
        biddable_projects = []
        for project in result['projects']:
            # Check if project is in a biddable state
            status = project.get('status')
            frontend_status = project.get('frontend_status')
            sub_status = project.get('sub_status')
            
            # Define states that allow bidding
            active_states = ['active', 'open', 'published']
            inactive_states = ['closed', 'frozen', 'cancelled']
            
            # Add to biddable projects if active or if status not provided
            if (status and status in active_states) or \
               (frontend_status and frontend_status in active_states) or \
               (not status and not frontend_status and not sub_status):
                biddable_projects.append(project)
        
        if not biddable_projects:
            print(f"{Fore.YELLOW}No biddable projects found matching the criteria.{Style.RESET_ALL}")
            return []
            
        # Display all found biddable projects
        print(f"{Fore.GREEN}Found {len(biddable_projects)} biddable projects:{Style.RESET_ALL}")
        print("-" * 80)
        
        for i, project in enumerate(biddable_projects):
            # Get budget info
            budget = project.get('budget', {})
            min_budget = budget.get('minimum', 'N/A')
            max_budget = budget.get('maximum', min_budget)
            
            if min_budget != 'N/A' and max_budget != 'N/A':
                budget_str = f"${min_budget}" if min_budget == max_budget else f"${min_budget}-${max_budget}"
            else:
                budget_str = "Budget not specified"
            
            # Get currency info
            currency = project.get('currency', {})
            currency_code = currency.get('code', 'USD')
            
            # Get job categories
            job_list = []
            for job in project.get('jobs', []):
                job_list.append(job.get('name', ''))
            job_str = ", ".join(job_list)
            
            # Get description snippet
            description = project.get('description', 'No description')
            if len(description) > 100:
                description_snippet = description[:97] + "..."
            else:
                description_snippet = description
            
            # Format date
            if 'time_submitted' in project:
                time_submitted = datetime.fromtimestamp(project['time_submitted'])
                date_str = time_submitted.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = "Unknown date"
            
            # Get owner info
            owner_id = project.get('owner_id')
            owner_info = result.get('users', {}).get(str(owner_id), {})
            owner_name = owner_info.get('display_name', 'Unknown')
            
            # Display location if available
            location = None
            if project.get('local') and project.get('location'):
                loc = project.get('location', {})
                city = loc.get('city', '')
                country = loc.get('country', {}).get('name', '')
                if city and country:
                    location = f"{city}, {country}"
                elif city:
                    location = city
                elif country:
                    location = country
            
            # Build project URL
            project_url = f"{session.url}/projects/{project.get('seo_url', project.get('id'))}"
            
            # Print project info
            print(f"{Fore.CYAN}[{i+1}] {Fore.WHITE}{project['title']}{Style.RESET_ALL} (ID: {project['id']})")
            print(f"    Posted: {date_str} by {owner_name}")
            print(f"    Budget: {budget_str} {currency_code}")
            if location:
                print(f"    Location: {location}")
            print(f"    Skills: {job_str}")
            print(f"    Description: {description_snippet}")
            print(f"    URL: {project_url}")
            print("-" * 80)
            
            # Save project details to file if requested
            if args.save_details:
                save_project_details(session, project['id'], args.output_dir)
        
        return biddable_projects
        
    except ProjectsNotFoundException as e:
        logger.error(f"Error finding projects: {e}")
        return []

def save_project_details(session, project_id, output_dir="project_details"):
    """Save full project details to a file"""
    try:
        # Get comprehensive project details
        project_details = create_get_projects_project_details_object(
            full_description=True,
            jobs=True,
            attachments=True,
            qualifications=True,
            location=True
        )
        
        user_details = create_get_projects_user_details_object(
            basic=True,
            profile_description=True,
            display_info=True,
            reputation=True
        )
        
        project = get_project_by_id(session, project_id, project_details, user_details)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format filename
        filename = f"{output_dir}/project_{project_id}.json"
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(project, f, indent=2)
            
        print(f"    {Fore.GREEN}Full details saved to {filename}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error saving project details: {e}")

#----- AI Analysis and Bidding Functions -----#

async def analyze_projects_with_ai(projects, job_categories, session, user_profile, user_skills, analyze_all=False, project_index=None):
    """
    Analyze projects using the AI agent.
    
    Args:
        projects: List of projects to analyze.
        job_categories: List of job categories.
        session: Freelancer API session.
        user_profile: User profile information.
        user_skills: List of user skills.
        analyze_all: Whether to analyze all projects or just one.
        project_index: Index of the project to analyze if analyze_all is False.
        
    Returns:
        The AI analysis result.
    """
    if not projects:
        print(f"{Fore.YELLOW}No projects to analyze.{Style.RESET_ALL}")
        return None
        
    if not OPENAI_API_KEY:
        print(f"{Fore.RED}Error: OPENAI_API_KEY environment variable not set.{Style.RESET_ALL}")
        print("Please set your OpenAI API key to use the AI analysis feature.")
        return None
    
    # Set up dependencies for the AI agent
    deps = SearchAgentDeps(
        projects=projects,
        job_categories=job_categories,
        session=session,
        user_profile=user_profile,
        user_skills=user_skills
    )
    
    # Determine the prompt based on whether we're analyzing all projects or just one
    if analyze_all:
        prompt = (
            f"Please analyze these {len(projects)} projects from Freelancer.com and provide insights about the market, "
            f"common skills, and recommendations. Consider my skills ({', '.join(user_skills)}) when evaluating projects. "
            f"Identify the top 3 projects that I should bid on based on match with my skills, budget, and project clarity."
        )
    else:
        if project_index is None or project_index < 1 or project_index > len(projects):
            print(f"{Fore.RED}Invalid project index. Please choose between 1 and {len(projects)}.{Style.RESET_ALL}")
            return None
            
        project = projects[project_index - 1]
        prompt = (
            f"Please analyze this project titled '{project.get('title')}' and provide a detailed summary and recommendation. "
            f"Consider my skills ({', '.join(user_skills)}) when evaluating if I should bid on this project."
        )
    
    print(f"\n{Fore.CYAN}Analyzing projects with AI...{Style.RESET_ALL}")
    
    # Run the AI agent
    result = await freelancer_agent.run(prompt, deps=deps)
    return result.data

async def generate_bid_proposal(project, session, user_profile, user_skills):
    """
    Generate a bid proposal for a specific project using the AI bidder agent.
    
    Args:
        project: The project to bid on.
        session: Freelancer API session.
        user_profile: User profile information.
        user_skills: List of user skills.
        
    Returns:
        A BidProposal object.
    """
    if not OPENAI_API_KEY:
        print(f"{Fore.RED}Error: OPENAI_API_KEY environment variable not set.{Style.RESET_ALL}")
        print("Please set your OpenAI API key to use the AI bidding feature.")
        return None
    
    # Set up dependencies for the bidder agent
    deps = BidderAgentDeps(
        project=project,
        session=session,
        user_profile=user_profile,
        user_skills=user_skills
    )
    
    # Create prompt for the bidder agent
    prompt = (
        f"Create a bid proposal for this project titled '{project.get('title')}'. "
        f"The project description is: '{project.get('description')}'. "
        f"My skills are: {', '.join(user_skills)}. "
        f"Make the proposal compelling and tailored to this specific project."
    )
    
    print(f"\n{Fore.CYAN}Generating bid proposal with AI...{Style.RESET_ALL}")
    
    # Run the bidder agent
    result = await bidder_agent.run(prompt, deps=deps)
    return result.data

# async def check_project_status(session, project_id):
#     """
#     Check if a project is in an active state that allows bidding.
    
#     Args:
#         session: Freelancer API session.
#         project_id: ID of the project to check.
        
#     Returns:
#         Boolean indicating if the project is biddable.
#     """
#     try:
#         # Get project details
#         project_details = create_get_projects_project_details_object(
#             full_description=True
#         )
        
#         project = get_project_by_id(
#             session,
#             project_id,
#             project_details
#         )
        
#         # Check if project exists
#         if not project:
#             logger.info(f"Project {project_id} not found.")
#             return False
            
#         # Check project status
#         # The exact field and values may vary depending on the API
#         if 'status' in project and project['status'] in ['active', 'open']:
#             return True
#         elif 'frontend_status' in project and project['frontend_status'] in ['active', 'open']:
#             return True
#         else:
#             logger.info(f"Project {project_id} is not in an active state for bidding.")
#             return False
            
#     except Exception as e:
#         logger.error(f"Error checking project status: {e}")
#         return False

async def check_project_status(session, project_id):
    """
    Check if a project is in an active state that allows bidding.
    
    Args:
        session: Freelancer API session.
        project_id: ID of the project to check.
        
    Returns:
        Boolean indicating if the project is biddable.
    """
    try:
        # Get project details
        project_details = create_get_projects_project_details_object(
            full_description=True
        )
        
        project = get_project_by_id(
            session,
            project_id,
            project_details
        )
        
        # Check if project exists
        if not project:
            logger.info(f"Project {project_id} not found.")
            return False
            
        # Log the project status to understand the structure
        logger.info(f"Project status: {project.get('status')}, frontend status: {project.get('frontend_status')}")
        
        # Try different status fields that might exist
        status = project.get('status')
        frontend_status = project.get('frontend_status')
        sub_status = project.get('sub_status')
        
        # Define states that allow bidding (may need adjustment based on actual API responses)
        active_states = ['active', 'open', 'published']
        
        # Check if any status field indicates an active state
        if (status and status in active_states) or \
           (frontend_status and frontend_status in active_states) or \
           (not status and not frontend_status):  # If status not provided, assume active
            return True
        else:
            logger.info(f"Project {project_id} is not in an active state for bidding.")
            return False
            
    except Exception as e:
        logger.error(f"Error checking project status: {e}")
        return False

async def submit_bid(session, proposal):
    """
    Submit a bid to a project on Freelancer.com.
    
    Args:
        session: Freelancer API session.
        proposal: BidProposal object.
        
    Returns:
        The result of the bid submission.
    """
    try:
        # Get project details to ensure it's valid
        project_id = proposal.project_id
        
        # Check if project is biddable
        is_biddable = await check_project_status(session, project_id)
        if not is_biddable:
            print(f"{Fore.YELLOW}This project is not available for bidding.{Style.RESET_ALL}")
            return None
        
        # Get current user's ID
        from freelancersdk.resources.users import get_self_user_id
        bidder_id = get_self_user_id(session)
        
        # Submit the bid
        bid_result = place_project_bid(
            session,
            project_id=project_id,
            bidder_id=bidder_id,
            amount=proposal.bid_amount,
            period=proposal.period,
            description=proposal.description,
            milestone_percentage=proposal.milestone_percentage
        )
            
        return bid_result
        
    except Exception as e:
        logger.error(f"Error submitting bid: {e}")
        print(f"{Fore.RED}Detailed error: {str(e)}{Style.RESET_ALL}")
        return None

def display_ai_analysis(analysis):
    """Display the AI analysis in a formatted way."""
    if not analysis:
        return
        
    print("\n" + "=" * 80)
    print(f"{Fore.GREEN}AI ANALYSIS RESULTS{Style.RESET_ALL}")
    print("=" * 80)
    
    if isinstance(analysis, ProjectSummary):
        print(f"{Fore.CYAN}PROJECT SUMMARY:{Style.RESET_ALL}")
        print(f"Title: {analysis.title}")
        print(f"Description: {analysis.description}")
        print(f"Skills Required: {analysis.skills_required}")
        print(f"Budget Range: {analysis.budget_range}")
        print(f"Recommendation: {analysis.recommendation}")
        print(f"Bid Confidence Score: {analysis.bid_confidence_score}/100")
    
    elif isinstance(analysis, ProjectAnalysis):
        print(f"{Fore.CYAN}MARKET ANALYSIS:{Style.RESET_ALL}")
        print(f"Average Budget: {analysis.average_budget}")
        print(f"Common Skills: {', '.join(analysis.common_skills)}")
        print(f"Market Insight: {analysis.market_insight}")
        print(f"Overall Recommendation: {analysis.recommendation}")
        
        if analysis.top_projects_to_bid:
            print(f"\n{Fore.GREEN}TOP RECOMMENDED PROJECTS TO BID ON:{Style.RESET_ALL}")
            for index in analysis.top_projects_to_bid:
                print(f"Project #{index}")
        
        print(f"\n{Fore.CYAN}PROJECT SUMMARIES:{Style.RESET_ALL}")
        for i, summary in enumerate(analysis.project_summaries):
            bid_confidence = summary.bid_confidence_score
            confidence_color = Fore.GREEN if bid_confidence >= 70 else Fore.YELLOW if bid_confidence >= 40 else Fore.RED
            
            print(f"\n{Fore.YELLOW}Project {i+1}: {summary.title}{Style.RESET_ALL}")
            print(f"Description: {summary.description}")
            print(f"Skills: {summary.skills_required}")
            print(f"Budget: {summary.budget_range}")
            print(f"Recommendation: {summary.recommendation}")
            print(f"Bid Confidence: {confidence_color}{bid_confidence}/100{Style.RESET_ALL}")
    
    print("=" * 80)

def display_bid_proposal(proposal):
    """Display the bid proposal in a formatted way."""
    if not proposal:
        return
        
    print("\n" + "=" * 80)
    print(f"{Fore.GREEN}BID PROPOSAL{Style.RESET_ALL}")
    print("=" * 80)
    
    print(f"{Fore.CYAN}Project ID:{Style.RESET_ALL} {proposal.project_id}")
    print(f"{Fore.CYAN}Bid Amount:{Style.RESET_ALL} ${proposal.bid_amount}")
    print(f"{Fore.CYAN}Delivery Period:{Style.RESET_ALL} {proposal.period} days")
    print(f"{Fore.CYAN}Milestone Percentage:{Style.RESET_ALL} {proposal.milestone_percentage}%")
    print(f"\n{Fore.CYAN}Proposal:{Style.RESET_ALL}")
    print(proposal.description)
    
    print("=" * 80)

def display_bid_result(result):
    """Display the bid submission result in a formatted way."""
    if not result:
        return
        
    print("\n" + "=" * 80)
    print(f"{Fore.GREEN}BID SUBMISSION RESULT{Style.RESET_ALL}")
    print("=" * 80)
    
    status = result.get('status', 'unknown')
    if status == 'simulated':
        print(f"{Fore.YELLOW}BID SIMULATION SUCCESSFUL{Style.RESET_ALL}")
        print(f"Bid amount: ${result.get('amount')}")
        print(f"Simulated bid ID: {result.get('id')}")
    elif status == 'success':
        print(f"{Fore.GREEN}BID SUCCESSFULLY SUBMITTED!{Style.RESET_ALL}")
        print(f"Bid amount: ${result.get('amount')}")
        print(f"Bid ID: {result.get('id')}")
    else:
        print(f"{Fore.RED}BID SUBMISSION FAILED{Style.RESET_ALL}")
        print(f"Status: {status}")
        print(f"Details: {result}")
    
    print("=" * 80)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Freelancer.com Bidder Agent")
    parser.add_argument('--query', type=str, help="Search query for projects")
    parser.add_argument('--job-ids', nargs='+', type=int, help="Job category IDs to filter by")
    parser.add_argument('--min-price', type=float, help="Minimum project budget")
    parser.add_argument('--max-price', type=float, help="Maximum project budget")
    parser.add_argument('--fixed-only', action='store_true', help="Only show fixed-price projects")
    parser.add_argument('--include-contests', action='store_true', help="Include contests in search")
    parser.add_argument('--sort', type=str, default='bid_count', 
                       help="Sort field (bid_count, time_updated, etc.)")
    parser.add_argument('--limit', type=int, default=10, help="Maximum number of projects to return")
    parser.add_argument('--analyze-all', action='store_true', help="Analyze all found projects")
    parser.add_argument('--analyze', type=int, help="Analyze specific project by index")
    parser.add_argument('--bid', type=int, help="Bid on specific project by index")
    parser.add_argument('--output-dir', type=str, default="project_details",
                       help="Directory to save project details")
    parser.add_argument('--save-details', action='store_true', 
                       help="Save full project details to files")
    parser.add_argument('--simulate-bid', action='store_true',
                       help="Simulate bid submission without actually placing bids")
    return parser.parse_args()

async def main():
    """Main function to run the Freelancer bidder agent"""
    args = parse_args()
    
    # Create API session and check connection
    session = create_session()
    user_profile = check_connection(session)
    if not user_profile:
        return
        
    # Get user skills and job categories
    user_skills = get_user_skills(session)
    job_categories = list_job_categories(session)
    
    # Find projects based on search criteria
    projects = find_projects(session, args)
    if not projects:
        return
        
    # Run AI analysis if requested
    if args.analyze_all or args.analyze:
        analysis = await analyze_projects_with_ai(
            projects=projects,
            job_categories=job_categories,
            session=session,
            user_profile=user_profile,
            user_skills=user_skills,
            analyze_all=args.analyze_all,
            project_index=args.analyze
        )
        display_ai_analysis(analysis)
        
    # Generate and submit bid if requested
    if args.bid:
        if args.bid < 1 or args.bid > len(projects):
            print(f"{Fore.RED}Invalid project index. Please choose between 1 and {len(projects)}{Style.RESET_ALL}")
            return
            
        project = projects[args.bid - 1]
        proposal = await generate_bid_proposal(
            project=project,
            session=session,
            user_profile=user_profile,
            user_skills=user_skills
        )
        
        if proposal:
            display_bid_proposal(proposal)
            if input(f"\n{Fore.YELLOW}Submit this bid? (y/n): {Style.RESET_ALL}").lower() == 'y':
                bid_result = await submit_bid(session, proposal)
                display_bid_result(bid_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())