from typing import Dict, List, Optional, Any
import string

class ResponseParser:
    def __init__(self, task_config: Dict[str, Any] = None):

        # Validate task_config
        if task_config is None:
            raise ValueError("task_config cannot be None")
            
        # Check for required keys
        required_keys = ['valid_fc_tokens', 'valid_rs_tokens']
        for key in required_keys:
            if key not in task_config:
                raise ValueError(f"task_config missing required key: {key}")
        
        # Validate token lists are not empty
        if not task_config['valid_fc_tokens']:
            raise ValueError("valid_fc_tokens cannot be empty")
        if not task_config['valid_rs_tokens']:
            raise ValueError("valid_rs_tokens cannot be empty")

        self.task_config = task_config
        self.token_dict = self._initialize_token_dict()
        self.fc_indices = self._initialize_fc_indices()
        self.rs_indices = self._initialize_rs_indices()


    def _initialize_token_dict(self) -> Dict[str, str]:
        """Initialize the mapping dictionary for valid response tokens."""
        # Create base dictionary with default mappings
        token_dict = {chr(i): '_' for i in range(128) 
                     if chr(i).isalnum() or chr(i) in string.punctuation}
        
        # Add valid response tokens
        valid_tokens = self.task_config['valid_fc_tokens']
        for token in valid_tokens:
            token_dict[token] = token
            
        return token_dict

    def _initialize_fc_indices(self) -> Dict[str, int]:
        """
        Initialize forced choice indices based on valid tokens.
        
        Returns:
            Dictionary mapping tokens to indices
        """
        valid_tokens = self.task_config['valid_fc_tokens']
        
        # Create indices for valid tokens
        fc_indices = {token: i for i, token in enumerate(valid_tokens)}
        
        # Add special '_' token for invalid responses
        fc_indices['_'] = len(valid_tokens)
        
        return fc_indices

    def _initialize_rs_indices(self) -> Dict[str, int]:
        """
        Initialize response set indices based on valid tokens.
        
        Returns:
            Dictionary mapping token combinations to indices
        """
        valid_tokens = self.task_config['valid_rs_tokens']
        
        # Create indices for valid tokens
        rs_indices = {token: i for i, token in enumerate(valid_tokens)}
        
        # Add special '_' token for invalid responses
        rs_indices['_'] = len(valid_tokens)
        
        return rs_indices

    def lookup_char(self, char: str) -> str:
        """
        Look up character in token dictionary.
        
        Args:
            char: Character to look up
            
        Returns:
            Valid token or '_' if invalid
        """
        if char not in self.token_dict:
            self.token_dict[char] = '_'
        return self.token_dict[char]

    def sanitize_tokens(self, response: str) -> List[str]:
        """
        Sanitize tokens in response string.
        
        Args:
            response: Raw response string
            
        Returns:
            List of sanitized tokens
        """
        return [self.lookup_char(char) for char in response]
 
    
    def parse_fc_response(self, response: str) -> int:
        """Parse a response string into a forced choice index.
        
        Args:
            response: Raw response string from the model
            
        Returns:
            Index of the option.
        """
        sanitized = self.sanitize_tokens(response)

        if len(sanitized) > 0:
            return self.fc_indices[sanitized[0]]
        else:
            # Null character
            return self.fc_indices['_']

    def parse_rs_response(self, response: str) -> int:
        """Parse a response string into a response set index.
        
        Args:
            response: Raw response string from the model
            
        Returns:
            Index of the option.
        """
        sanitized = ''.join(sorted(self.sanitize_tokens(response)))

        if sanitized in self.rs_indices:
            return self.rs_indices[sanitized]
        else:
            return self.rs_indices['_']

    def parse_response(self, response: str, fmt: str) -> int:
        """
        Parse response based on format.

        Args:
            response: Raw response string
            fmt: Format type ('FC' or 'RS')
            
        Returns:
            Parsed response index
        """
        return self.parse_rs_response(response) if fmt == 'RS' else self.parse_fc_response(response)
