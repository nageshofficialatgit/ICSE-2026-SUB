// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.28;

interface IERC20 {
    /// @notice Returns the total supply of the token.
    function totalSupply() external view returns (uint256);

    /// @notice Returns the balance for the specified account.
    /// @param account The address of the account to query.
    function balanceOf(address account) external view returns (uint256);

    /// @notice Transfers an amount of the token to a recipient.
    /// @param recipient The address receiving the token.
    /// @param amount The amount of the token to transfer.
    function transfer(address recipient, uint256 amount)
        external
        returns (bool);

    /// @notice Returns the remaining amount of the token that spender can spend on behalf of owner.
    /// @param owner The address that owns the token.
    /// @param spender The address allowed to spend the token.
    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    /// @notice Approves spender to spend amount on behalf of the caller.
    /// @param spender The address allowed to spend the token.
    /// @param amount The amount of the token to approve.
    function approve(address spender, uint256 amount) external returns (bool);

    /// @notice Transfers an amount of the token from sender to recipient using the allowance mechanism.
    /// @param sender The address from which an amount of the token is transferred.
    /// @param recipient The address receiving the token.
    /// @param amount The amount of the token to transfer.
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    /// @notice Emitted when an amount of the token ('value') is transferred from one account to another.
    event Transfer(address indexed from, address indexed to, uint256 value);

    /// @notice Emitted when owner approves spender to spend an amount of the token ('value').
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
}


/// @title PYOConnectionManager
/// @notice Relays connection requests and responses
/// @dev Intended for in-app use
/// @dev Includes adjustable request fee for spam prevention
/// @author S.E.
/// @custom:additional-information https://github.com/sewing848/pyoconnectionmanager
contract PYOConnectionManager {

    /// @notice Address with privileged control over contract
    address public owner;

    /// @notice Mapping of addresses with administrative privileges
    mapping(address => bool) public isAdmin;

    /// @notice ERC20 token used for paying connection request fees
    IERC20 public token;

    /// @notice Amount of token required to send a connection request
    uint256 public requestFee;

    /// @notice Emergency pause of new connection requests
    bool public requestsPaused = false;
    
    /// @notice Emergency pause of new responses to connection requests
    bool public responsesPaused = false;
    
    /// @notice Emergency pause of withdrawals of request-fee tokens by admins
    bool public adminWithdrawalsPaused = false;


    // ------------------------------------------------------------------------
    //                     Events
    // ------------------------------------------------------------------------

    /// @notice Emitted when a connection request is sent
    /// @param to Address receiving the request
    /// @param from Address initiating the request
    /// @param publicKey The sender's publicKey, needed to connect
    /// @param data Encrypted connection data
    /// @param timestamp Block timestamp for transaction
    event PYOConnectionRequest(
        address indexed to,
        address from,
        uint256 publicKey,
        bytes data,
        uint256 timestamp
    );

    /// @notice Emitted when a connection response is sent
    /// @param to Address receiving the response
    /// @param from Address sending the response
    /// @param response Encrypted response to request
     /// @param timestamp Block timestamp for transaction
    event PYOConnectionResponse(
        address indexed to, 
        address indexed from,
        uint256 response,
        uint256 timestamp
    );

    /// @notice Emitted when the request pause state is changed
    /// @param admin Address of the admin changing the pause state
    /// @param newPauseState True if requests are paused, false if unpaused
    /// @param timestamp Block timestamp when the change occurred
    event RequestsPauseStateChanged(
        address admin,
        bool newPauseState,
        uint256 timestamp
    );

    /// @notice Emitted when the response pause state is changed
    /// @param admin Address of the admin changing the pause state
    /// @param newPauseState True if responses are paused, false if unpaused
    /// @param timestamp Block timestamp when the change occurred
    event ResponsesPauseStateChanged(
        address admin,
        bool newPauseState,
        uint256 timestamp
    );

    /// @notice Emitted when the admin withdrawal pause state is changed
    /// @param owner Address of the owner changing the pause state
    /// @param newPauseState True if withdrawals are paused, false if unpaused
    /// @param timestamp Block timestamp when the change occurred
    event AdminWithdrawalsPauseStateChanged(
        address owner,
        bool newPauseState,
        uint256 timestamp
    );

    /// @notice Emitted when the request fee is updated
    /// @param admin Address of the admin setting the new fee
    /// @param newFee The updated fee amount
    /// @param timestamp Block timestamp when the change occurred
    event RequestFeeChanged(
        address admin,
        uint256 newFee,
        uint256 timestamp
    );

    /// @notice Emitted when tokens are withdrawn from the contract
    /// @param tokenContract Address of the token contract
    /// @param admin Address of the admin withdrawing tokens
    /// @param amount Amount of tokens withdrawn
    /// @param timestamp Block timestamp when the withdrawal occurred
    event TokensWithdrawn(
        address tokenContract,
        address admin,
        uint256 amount,
        uint256 timestamp
    );

    /// @notice Emitted when the token used for fees is changed
    /// @param oldToken Address of the previous token
    /// @param newToken Address of the new token
    /// @param owner Address of the owner making the change
    /// @param timestamp Block timestamp when the change occurred
    event TokenAddressChanged(
        address oldToken,
        address newToken,
        address owner,
        uint256 timestamp
    );

    /// @notice Emitted when a new admin is added
    /// @param newAdmin Address of the new admin
    /// @param addedBy Address of the owner who added the admin
    /// @param timestamp Block timestamp when the change occurred
    event AdminAdded(
        address newAdmin,
        address addedBy,
        uint256 timestamp
    );

    /// @notice Emitted when an admin is removed
    /// @param admin Address of the removed admin
    /// @param removedBy Address of the owner who removed the admin
    /// @param timestamp Block timestamp when the change occurred
    event AdminRemoved(
        address admin,
        address removedBy,
        uint256 timestamp
    );

    /// @notice Emitted when an admin resigns from their role
    /// @param admin Address of the admin resigning
    /// @param timestamp Block timestamp when the resignation occurred
    event AdminResigned(
        address admin,
        uint256 timestamp
    );

    /// @notice Emitted when contract ownership is transferred
    /// @param oldOwner Address of the previous owner
    /// @param newOwner Address of the new owner
    /// @param timestamp Block timestamp when the transfer occurred
    event OwnershipTransferred(
        address oldOwner,
        address newOwner,
        uint256 timestamp
    );


    // ------------------------------------------------------------------------
    //                     Constructor
    // ------------------------------------------------------------------------

    /// @notice Sets the contract creator as owner
    /// @notice Establishes initial fee and contract address for fee token
    constructor() {
        token = IERC20(0x24796F5620Ac8B6e5059C033a7a6860B70D7098c);
        requestFee = 1e19;
        owner = msg.sender;
        // Set owner to be an admin as well
        isAdmin[owner] = true;

    }


    // ------------------------------------------------------------------------
    //                     Modifiers
    // ------------------------------------------------------------------------
    
    /// @notice Restricts function access to contract owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    /// @notice Restricts function access to admins
    modifier onlyAdmin() {
        require(isAdmin[msg.sender], "Not an admin");
        _;
    }
    
    /// @notice Restricts function access to admins or owner
    modifier onlyAdminOrOwner() {
        require(isAdmin[msg.sender] || msg.sender == owner, "Not admin or owner");
        _;
    }

    /// @notice Checks if requests are paused
    modifier whenRequestsNotPaused() {
        require(!requestsPaused, "Requests are paused");
        _;
    }

    /// @notice Checks if responses are paused
    modifier whenResponsesNotPaused() {
        require(!responsesPaused, "Responses are paused");
        _;
    }

    /// @notice Checks if admin withdrawals are paused
    modifier whenAdminWithdrawalsNotPaused() {
        // Owner can always withdraw regardless of pause state
        if (msg.sender != owner) {
            require(!adminWithdrawalsPaused, "Admin withdrawals are paused");
        }
        _;
    }


    // ------------------------------------------------------------------------
    //                     Administrative Functions
    // ------------------------------------------------------------------------

    /// @notice Allows owner to transfer ownership
    /// @param _newOwner Address of the new owner
    function transferOwnership(address _newOwner)
        external
        onlyOwner
    {
        require(_newOwner != address(0), "Cannot use zero address");
        address oldOwner = owner;
        owner = _newOwner;
        // New owner doesn't automatically become admin
        // Old owner keeps admin status if they had it
        emit OwnershipTransferred(oldOwner, owner, block.timestamp);
    }
    

    /// @notice Allows owner to add a new admin
    /// @param _admin Address of the new admin
    function addAdmin(address _admin)
        external
        onlyOwner
    {
        require(_admin != address(0), "Cannot use zero address");
        require(!isAdmin[_admin], "Already an admin");
        isAdmin[_admin] = true;
        emit AdminAdded(_admin, msg.sender, block.timestamp);
    }
    

    /// @notice Allows owner to remove an admin
    /// @param _admin Address of the admin to remove
    function removeAdmin(address _admin)
        external
        onlyOwner
    {
        require(isAdmin[_admin], "Not an admin");
        delete isAdmin[_admin];
        emit AdminRemoved(_admin, msg.sender, block.timestamp);
    }
    

    /// @notice Allows an admin to resign from their role
    function resignAsAdmin()
        external
        onlyAdmin
    {
        delete isAdmin[msg.sender];
        emit AdminResigned(msg.sender, block.timestamp);
    }


    /// @notice Allows owner or admin to withdraw tokens sent to the contract
    /// @param tokenAddress Address of the token to withdraw
    /// @param amount Amount to withdraw
    /// @param recipient Address to receive the tokens
    function withdrawTokens(address tokenAddress, uint256 amount, address recipient)
        external
        onlyAdminOrOwner
        whenAdminWithdrawalsNotPaused
    {
        require(recipient != address(0), "Cannot withdraw to zero address");
        require(amount > 0, "Amount must be greater than zero");
        require(IERC20(tokenAddress).balanceOf(address(this)) >= amount, "Insufficient balance");
        
        require(IERC20(tokenAddress).transfer(recipient, amount), "Token transfer failed");
        emit TokensWithdrawn(tokenAddress, recipient, amount, block.timestamp);
    }


    /// @notice Allows owner to change the token used for fees
    /// @param _newToken Address of the new token
    function setTokenAddress(address _newToken)
        external
        onlyOwner
    {
        require(_newToken != address(0), "Cannot use zero address");
        address oldToken = address(token);
        token = IERC20(_newToken);
        emit TokenAddressChanged(oldToken, _newToken, msg.sender, block.timestamp);
    }


    // ------------------------------------------------------------------------
    //                     Pause Control Functions
    // ------------------------------------------------------------------------

    /// @notice Allows admin to pause or unpause connection requests
    /// @param _paused New pause state
    function setRequestsPaused(bool _paused)
        external
        onlyAdmin
    {
        requestsPaused = _paused;
        emit RequestsPauseStateChanged(msg.sender, _paused, block.timestamp);
    }


    /// @notice Allows admin to pause or unpause connection responses
    /// @param _paused New pause state
    function setResponsesPaused(bool _paused)
        external
        onlyAdmin
    {
        responsesPaused = _paused;
        emit ResponsesPauseStateChanged(msg.sender, _paused, block.timestamp);
    }


    /// @notice Allows owner to pause or unpause admin withdrawals
    /// @param _paused New pause state
    function setAdminWithdrawalsPaused(bool _paused)
        external
        onlyOwner
    {
        adminWithdrawalsPaused = _paused;
        emit AdminWithdrawalsPauseStateChanged(msg.sender, _paused, block.timestamp);
    }


    // ------------------------------------------------------------------------
    //                      Request and Response Functions
    // ------------------------------------------------------------------------

    /// @notice Allows admin to update the request fee
    /// @param _newFee New fee amount
    function setRequestFee(uint256 _newFee)
        external
        onlyAdmin
    {
        requestFee = _newFee;
        emit RequestFeeChanged(msg.sender, requestFee, block.timestamp);
    }


    /// @notice Sends a connection request to another user
    /// @dev Token fee required for requests unless set to 0
    /// @param to Address receiving the request
    /// @param publicKey The sender's public key
    /// @param data Encrypted connection data
    function sendConnectionRequest(address to, uint256 publicKey, bytes calldata data) 
        external
        whenRequestsNotPaused
    {
        require(to != address(0), "Cannot use zero address");
        require(to != msg.sender, "Cannot send to self");
    
    // Only perform token transfer if the request fee is greater than 0
        if (requestFee > 0) {
            require(token.transferFrom(msg.sender, address(this), requestFee), 
                "Token transfer failed");
        }
    
    // Emit connection request event
        emit PYOConnectionRequest(to, msg.sender, publicKey, data, block.timestamp);
    }


    /// @notice Sends a response to a connection request
    /// @dev No token fee required for responses
    /// @param to Address that sent the original request
    /// @param response The sender's encrypted response to a request
    function sendConnectionResponse(
        address to, 
        uint256 response
    ) 
        external
        whenResponsesNotPaused
    {
        require(to != address(0), "Cannot use zero address");
        require(to != msg.sender, "Cannot send to self");
        
        emit PYOConnectionResponse(to, msg.sender, response, block.timestamp);
    }

}


/*

 Copyright 2025 S.E.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/