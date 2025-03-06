#!/usr/bin/env python3
"""
Safe test script to search projects on Freelancer.com (no bidding by default)
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from freelancersdk.session import Session
from freelancersdk.resources.projects import (
    search_projects, place_project_bid, get_jobs,
    create_search_projects_filter
)
from freelancersdk.resources.users import get_self_user_id, get_self
from freelancersdk.exceptions import (
    ProjectsNotFoundException, BidNotPlacedException
)

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
FLN_URL = os.getenv("FLN_URL", "https://www.freelancer.com")
OAUTH_TOKEN = os.getenv("FLN_OAUTH_TOKEN")

# Bid parameters (only used if --place-bid flag is provided)
BID_AMOUNT = 100  # Bid amount in USD
BID_PERIOD = 7    # Delivery time in days
BID_DESCRIPTION = """
Hello! I'm interested in working on your project. I have extensive experience in this area and can deliver high-quality results within your specified timeframe.

With my expertise in:
- Python development
- API integration
- Data analysis

I'm confident I can exceed your expectations. If you have any questions, please let me know.

Looking forward to your response!
"""

def create_session():
    """Create a session with the Freelancer API"""
    if not OAUTH_TOKEN:
        print("ERROR: FLN_OAUTH_TOKEN not found in environment variables")
        sys.exit(1)
        
    print(f"Connecting to {FLN_URL}...")
    return Session(oauth_token=OAUTH_TOKEN, url=FLN_URL)

def check_connection(session):
    """Check if we can connect to the API and get basic user info"""
    try:
        user_id = get_self_user_id(session)
        user_info = get_self(session)
        
        print(f"Successfully connected to Freelancer API!")
        print(f"User ID: {user_id}")
        print(f"Username: {user_info.get('username', 'Unknown')}")
        print(f"Display name: {user_info.get('display_name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"Error connecting to Freelancer API: {e}")
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
            
    except Exception as e:
        print(f"Error retrieving job categories: {e}")

def find_projects(session, args):
    """Find suitable projects to potentially bid on"""
    print("\nSearching for projects...")
    
    # Get job IDs from args or use defaults
    job_ids = args.job_ids if args.job_ids else [1, 13, 32]  # Default: Python, Web, API
    
    # Set up search parameters
    search_filter = create_search_projects_filter(
        project_types=["fixed"] if args.fixed_only else None,
        min_avg_price=args.min_price,
        max_avg_price=args.max_price,
        sort_field=args.sort,
        jobs=job_ids
    )
    
    try:
        # Search for projects
        result = search_projects(
            session,
            query=args.query,
            search_filter=search_filter,
            limit=args.limit
        )
        
        if not result or not result['projects']:
            print("No projects found matching the criteria.")
            return []
            
        # Display all found projects
        projects = result['projects']
        print(f"Found {len(projects)} projects:")
        print("-" * 80)
        
        for i, project in enumerate(projects):
            print(project)
            # Get budget info
            min_budget = project.get('budget', {}).get('minimum', 'N/A')
            max_budget = project.get('budget', {}).get('maximum', min_budget)
            
            if min_budget != 'N/A' and max_budget != 'N/A':
                budget_str = f"${min_budget}" if min_budget == max_budget else f"${min_budget}-${max_budget}"
            else:
                budget_str = "Budget not specified"
            
            # Get description snippet
            description = project.get('preview_description', 'No description')
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
            
            # Print project info
            print(f"[{i+1}] {project['title']} (ID: {project['id']})")
            print(f"    Posted: {date_str}")
            print(f"    Budget: {budget_str}")
            print(f"    Description: {description_snippet}")
            print("-" * 80)
            
            # Save project details to file if requested
            if args.save_details:
                save_project_details(project)
        
        return projects
        
    except ProjectsNotFoundException as e:
        print(f"Error finding projects: {e}")
        return []

def save_project_details(project):
    """Save full project details to a file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs("project_details", exist_ok=True)
        
        # Format filename
        filename = f"project_details/project_{project['id']}.json"
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(project, f, indent=2)
            
        print(f"    Full details saved to {filename}")
    except Exception as e:
        print(f"    Error saving project details: {e}")

def place_bid(session, project_id):
    """Place a bid on a project"""
    print(f"Preparing to place bid on project {project_id}...")
    print("WARNING: THIS WILL PLACE A REAL BID ON A REAL PROJECT.")
    print(f"Bid amount: ${BID_AMOUNT}")
    print(f"Delivery period: {BID_PERIOD} days")
    print(f"Description: {BID_DESCRIPTION}...")
    
    confirm = input("\nType 'CONFIRM' to proceed or anything else to cancel: ")
    
    if confirm != "CONFIRM":
        print("Bid cancelled.")
        return None
    
    print(f"Placing bid on project {project_id}...")
    
    try:
        # Get user ID
        user_id = get_self_user_id(session)
        
        # Prepare bid data
        bid_data = {
            'project_id': project_id,
            'bidder_id': user_id,
            'amount': BID_AMOUNT,
            'period': BID_PERIOD,
            'milestone_percentage': 100,
            'description': BID_DESCRIPTION,
        }
        
        # Place the bid
        bid = place_project_bid(session, **bid_data)
        
        print("Bid placed successfully!")
        print(f"Bid ID: {bid.id}")
        print(f"Amount: ${bid.amount}")
        print(f"Period: {bid.period} days")
        
        return bid
        
    except BidNotPlacedException as e:
        print(f"Error placing bid: {e}")
        return None

def main():
    """Main function"""
    print("=" * 80)
    print("=== Freelancer.com Project Search and Bid Tool ===")
    print("=" * 80)
    
    # Add command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Freelancer.com API Test Tool")
    
    # Search parameters
    parser.add_argument("--query", type=str, default="Python API", help="Search query for projects")
    parser.add_argument("--min-price", type=int, default=50, help="Minimum project budget")
    parser.add_argument("--max-price", type=int, default=500, help="Maximum project budget")
    parser.add_argument("--limit", type=int, default=5, help="Number of projects to retrieve")
    parser.add_argument("--fixed-only", action="store_true", help="Show only fixed-price projects")
    parser.add_argument("--sort", type=str, default="time_updated", choices=["time_updated", "time_submitted"], 
                        help="Sort field for projects")
    parser.add_argument("--job-ids", type=int, nargs="+", help="Job category IDs to filter by")
    parser.add_argument("--save-details", action="store_true", help="Save full project details to files")
    
    # Actions
    parser.add_argument("--list-jobs", action="store_true", help="List available job categories")
    parser.add_argument("--place-bid", action="store_true", help="Enable bid placement (USE WITH CAUTION)")
    
    args = parser.parse_args()
    
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
    if not projects:
        return
    
    # Only proceed to bid placement if explicitly enabled
    if args.place_bid:
        # Ask which project to bid on
        try:
            selection = int(input(f"\nEnter project number to bid on (1-{len(projects)}) or 0 to cancel: "))
            if selection <= 0 or selection > len(projects):
                print("No bid will be placed. Exiting.")
                return
            
            # Get selected project
            project = projects[selection-1]
            
            # Double check
            print(f"\nYou selected: {project['title']}")
            budget_min = project.get('budget', {}).get('minimum', 'N/A')
            budget_max = project.get('budget', {}).get('maximum', budget_min)
            budget_str = f"${budget_min}" if budget_min == budget_max else f"${budget_min}-${budget_max}"
            print(f"Budget: {budget_str}")
            
            # Place the bid
            place_bid(session, project['id'])
        
        except ValueError:
            print("Invalid selection. No bid will be placed.")
    else:
        print("\nBid placement is disabled. Run with --place-bid flag to enable bidding.")
        print("CAUTION: Enabling bidding will place REAL bids on REAL projects!")

if __name__ == "__main__":
    main()