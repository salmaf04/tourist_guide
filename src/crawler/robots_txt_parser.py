from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class RobotsTxtInfo:
    """Información extraída de robots.txt"""
    allowed: List[str]
    disallowed: List[str]
    crawl_delay: float
    request_rate: Optional[str]

class RobotsTxtParser:
    """Parser para archivos robots.txt"""
    
    @staticmethod
    def parse(robots_txt_content: str) -> RobotsTxtInfo:
        """Parsea el contenido de robots.txt"""
        lines = robots_txt_content.split('\n')
        allowed = []
        disallowed = []
        crawl_delay = 1.0
        request_rate = None
        
        current_user_agent = None
        applies_to_us = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.lower().startswith('user-agent:'):
                user_agent = line.split(':', 1)[1].strip()
                applies_to_us = user_agent == '*' or 'tourismcrawler' in user_agent.lower()
                current_user_agent = user_agent
                
            elif applies_to_us:
                if line.lower().startswith('disallow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        disallowed.append(path)
                        
                elif line.lower().startswith('allow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        allowed.append(path)
                        
                elif line.lower().startswith('crawl-delay:'):
                    try:
                        crawl_delay = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        crawl_delay = 1.0
                        
                elif line.lower().startswith('request-rate:'):
                    request_rate = line.split(':', 1)[1].strip()
        
        return RobotsTxtInfo(
            allowed=allowed,
            disallowed=disallowed,
            crawl_delay=crawl_delay,
            request_rate=request_rate
        )