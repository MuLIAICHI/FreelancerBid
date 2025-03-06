#!/usr/bin/env python3
"""
Enhanced Freelancer.com Project Search AI Agent

This script enhances the Freelancer.com Project Search Tool with an AI agent
that can interpret natural language queries and provide intelligent responses
about project search results.

Requirements:
- freelancersdk package (pip install freelancersdk)
- Python-dotenv (pip install python-dotenv)
- pydantic-ai-slim (pip install 'pydantic-ai-slim[openai]')

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

# Define models for our AI agent
class ProjectSummary(BaseModel):
    """A summary of a Freelancer.com project."""
    title: str = Field(description="The title of the project")
    description: str = Field(description="Brief summary of the project's key points")
    skills_required: str = Field(description="Skills required for the project")
    budget_range: str = Field(description="The budget range for the project")
    recommendation: str = Field(description="Recommendation on whether this is a good project to bid on")

class ProjectAnalysis(BaseModel):
    """Analysis of multiple projects from Freelancer.com."""
    project_summaries: List[ProjectSummary] = Field(description="Summaries of individual projects")
    common_skills: List[str] = Field(description="Common skills requested across projects")
    average_budget: str = Field(description="Average budget across all projects")
    market_insight: str = Field(description="Insight about the current market based on these projects")
    recommendation: str = Field(description="Overall recommendation based on the projects analyzed")

class BidProposal(BaseModel):
    """A bid proposal for a Freelancer.com project."""
    project_id: int = Field(description="ID of the project to bid on")
    bid_amount: float = Field(description="Amount to bid in the project's currency")
    period: int = Field(description="Estimated completion time in days")
    description: str = Field(description="Bid proposal text")
    milestone_percentage: int = Field(description="Percentage of the bid amount for the first milestone (0-100)", ge=0, le=100)


@dataclass
class SearchAgentDeps:
    """Dependencies for the Freelancer search agent."""
    projects: List[Dict[str, Any]]
    job_categories: List[Dict[str, Any]]
    session: Session


@dataclass
class BidderAgentDeps:
    """Dependencies for the Freelancer bidder agent."""
    project: Dict[str, Any]
    session: Session
    user_profile: Dict[str, Any]
    user_skills: List[str]
    bid_history: List[Dict[str, Any]] = None
    typical_bid_amount: float = None

# Create our AI agent
freelancer_agent = Agent[SearchAgentDeps, Union[ProjectSummary, ProjectAnalysis]](
    'openai:gpt-4o',
    deps_type=SearchAgentDeps,
    result_type=Union[ProjectSummary, ProjectAnalysis],
    system_prompt=(
        "You are an expert freelancer assistant who helps users analyze projects from Freelancer.com. "
        "Your goal is to help the user identify good projects to bid on and provide insights about the market. "
        "When analyzing projects, consider the following: clarity of project description, budget reasonableness, "
        "client's history if available, and required skills."
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
        return True
    except Exception as e:
        logger.error(f"Error connecting to Freelancer API: {e}")
        return False

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
            limit=args.limit
        )
        
        if not result or not result['projects']:
            print(f"{Fore.YELLOW}No projects found matching the criteria.{Style.RESET_ALL}")
            return []
            
        # Display all found projects
        projects = result['projects']
        print(f"{Fore.GREEN}Found {len(projects)} projects:{Style.RESET_ALL}")
        print("-" * 80)
        
        for i, project in enumerate(projects):
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
        
        return projects
        
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

async def analyze_projects_with_ai(projects, job_categories, session, analyze_all=False, project_index=None):
    """
    Analyze projects using the AI agent.
    
    Args:
        projects: List of projects to analyze.
        job_categories: List of job categories.
        session: Freelancer API session.
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
        session=session
    )
    
    # Determine the prompt based on whether we're analyzing all projects or just one
    if analyze_all:
        prompt = f"Please analyze these {len(projects)} projects from Freelancer.com and provide insights about the market, common skills, and recommendations."
    else:
        if project_index is None or project_index < 1 or project_index > len(projects):
            print(f"{Fore.RED}Invalid project index. Please choose between 1 and {len(projects)}.{Style.RESET_ALL}")
            return None
            
        project = projects[project_index - 1]
        prompt = f"Please analyze this project titled '{project.get('title')}' and provide a detailed summary and recommendation."
    
    print(f"\n{Fore.CYAN}Analyzing projects with AI...{Style.RESET_ALL}")
    
    # Run the AI agent
    result = await freelancer_agent.run(prompt, deps=deps)
    return result.data

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
    
    elif isinstance(analysis, ProjectAnalysis):
        print(f"{Fore.CYAN}MARKET ANALYSIS:{Style.RESET_ALL}")
        print(f"Average Budget: {analysis.average_budget}")
        print(f"Common Skills: {', '.join(analysis.common_skills)}")
        print(f"Market Insight: {analysis.market_insight}")
        print(f"Overall Recommendation: {analysis.recommendation}")
        
        print(f"\n{Fore.CYAN}PROJECT SUMMARIES:{Style.RESET_ALL}")
        for i, summary in enumerate(analysis.project_summaries):
            print(f"\n{Fore.YELLOW}Project {i+1}: {summary.title}{Style.RESET_ALL}")
            print(f"Description: {summary.description}")
            print(f"Skills: {summary.skills_required}")
            print(f"Budget: {summary.budget_range}")
            print(f"Recommendation: {summary.recommendation}")
    
    print("=" * 80)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Freelancer.com Project Search AI Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Search parameters
    parser.add_argument("--query", type=str, default="", 
                        help="Search query for projects")
    parser.add_argument("--min-price", type=int, default=0, 
                        help="Minimum project budget")
    parser.add_argument("--max-price", type=int, default=None, 
                        help="Maximum project budget")
    parser.add_argument("--limit", type=int, default=10, 
                        help="Number of projects to retrieve")
    parser.add_argument("--fixed-only", action="store_true", 
                        help="Show only fixed-price projects")
    parser.add_argument("--include-contests", action="store_true", 
                        help="Include contests in search results")
    parser.add_argument("--sort", type=str, default="time_updated", 
                        choices=["time_updated", "time_submitted", "bid_count"], 
                        help="Sort field for projects")
    parser.add_argument("--job-ids", type=int, nargs="+", 
                        help="Job category IDs to filter by")
    parser.add_argument("--save-details", action="store_true", 
                        help="Save full project details to files")
    parser.add_argument("--output-dir", type=str, default="project_details",
                        help="Directory to save project details")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    
    # Actions
    parser.add_argument("--list-jobs", action="store_true", 
                        help="List available job categories")
    
    # AI analysis options
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze all search results with AI")
    parser.add_argument("--analyze-project", type=int,
                        help="Analyze a specific project with AI (specify index)")
    
    return parser.parse_args()

async def main():
    """Main function"""
    print("=" * 80)
    print(f"{Fore.CYAN}=== Freelancer.com Project Search AI Agent ==={Style.RESET_ALL}")
    print("=" * 80)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create session
    session = create_session()
    
    # Check connection
    if not check_connection(session):
        return
    
    # List job categories if requested
    job_categories = list_job_categories(session) if args.list_jobs else get_jobs(
        session, job_ids=[], seo_details=True, lang='en'
    )
    
    if args.list_jobs:
        return
    
    # Find projects
    projects = find_projects(session, args)
    
    # AI analysis if requested
    analysis = None
    if args.analyze and projects:
        analysis = await analyze_projects_with_ai(projects, job_categories, session, analyze_all=True)
    elif args.analyze_project is not None and projects:
        analysis = await analyze_projects_with_ai(projects, job_categories, session, 
                                         analyze_all=False, project_index=args.analyze_project)
    
    # Display AI analysis if available
    if analysis:
        display_ai_analysis(analysis)
    
    # Summary
    if projects:
        print(f"{Fore.GREEN}Found {len(projects)} projects matching your criteria.{Style.RESET_ALL}")
        if args.save_details:
            print(f"Detailed information saved to {args.output_dir}/")
    else:
        print(f"{Fore.YELLOW}No projects found matching your criteria.{Style.RESET_ALL}")
        print("Try adjusting your search parameters or increasing the limit.")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)