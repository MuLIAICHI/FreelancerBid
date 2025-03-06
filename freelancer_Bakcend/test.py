#!/usr/bin/env python3
"""
Enhanced Freelancer.com Project Search Tool

This script searches for projects on Freelancer.com based on specified criteria
and displays them in a readable format. It can also save detailed project information
to files for later analysis.

Requirements:
- freelancersdk package (pip install freelancersdk)
- Python-dotenv (pip install python-dotenv)

Configuration:
Create a .env file with the following variables:
FLN_URL=https://www.freelancer.com (or https://www.freelancer-sandbox.com for testing)
FLN_OAUTH_TOKEN=your_oauth_token_here
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
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

# Initialize colorama for cross-platform colored terminal output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_session():
    """Create a session with the Freelancer API"""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment variables
    fln_url = os.getenv("FLN_URL", "https://www.freelancer.com")
    oauth_token = os.getenv("FLN_OAUTH_TOKEN")
    
    if not oauth_token:
        logger.error("ERROR: FLN_OAUTH_TOKEN not found in environment variables")
        sys.exit(1)
        
    logger.info(f"Connecting to {fln_url}...")
    return Session(oauth_token=oauth_token, url=fln_url)

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Freelancer.com Project Search Tool",
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
    
    return parser.parse_args()

def main():
    """Main function"""
    print("=" * 80)
    print(f"{Fore.CYAN}=== Freelancer.com Project Search Tool ==={Style.RESET_ALL}")
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
    if args.list_jobs:
        list_job_categories(session)
        return
    
    # Find projects
    projects = find_projects(session, args)
    
    # Summary
    if projects:
        print(f"{Fore.GREEN}Found {len(projects)} projects matching your criteria.{Style.RESET_ALL}")
        if args.save_details:
            print(f"Detailed information saved to {args.output_dir}/")
    else:
        print(f"{Fore.YELLOW}No projects found matching your criteria.{Style.RESET_ALL}")
        print("Try adjusting your search parameters or increasing the limit.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)