// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/**
 * @title Basic Proxy Contract
 *
 * This contract is supported by HypeLoot.com
 * Stay connected with us:
 * - Website: https://hypeloot.com/
 * - X (formerly Twitter): https://x.com/HypeLootCom
 * - Telegram Channel: https://t.me/Hypelootcom
 * - Instagram: https://instagram.com/hypelootcom
 *
 * For platform support, please email: support[at]hypeloot.com
 * We are continuously expanding our activities.
 *
 * @notice 
 *   1. This contract implements a proxy pattern, where function calls are delegated via `delegatecall`
 *      to another contract (the "implementation" or "logic" contract). 
 *      This setup decouples the contract’s state (stored in the proxy) from the contract’s logic 
 *      (executed in the implementation).
 *
 *   2. The proxy stores two primary data points in dedicated storage slots:
 *      - `proxyOwner`: An address with the authorization to manage critical functions, such as 
 *        updating the implementation address or transferring proxy ownership. 
 *      - `implementation`: The address of the logic contract to which the proxy delegates calls.
 *      These dedicated storage slots help avoid storage conflicts between the proxy and the 
 *      implementation.
 *
 *   3. Additionally, this proxy records the following information:
 *      - The contract deployment timestamp (`_DEPLOYED_AT`), providing a reliable reference 
 *        for when the proxy was first deployed on-chain.
 *      - A hash of the contract name (constructed in the constructor), allowing on-chain tracking 
 *        or verification of the name assigned at deployment.
 *
 *   4. One of the key benefits of a proxy contract is upgradability. Via this proxy, the `proxyOwner` 
 *      can:
 *        - Update the `implementation` address, effectively upgrading or changing the business logic 
 *          without altering the proxy’s address or resetting its stored state.
 *        - Transfer the `proxyOwner` role to another address, enabling decentralized administrative 
 *          control or delegation to a governance mechanism over time.
 *
 *   5. To ensure transparency and track changes over time, the proxy also stores the previous 
 *      implementation address in a dedicated storage slot. The contract exposes functions to:
 *        - Retrieve the current implementation address.
 *        - Retrieve the previous implementation address (to facilitate upgrade audits or potential 
 *          rollback strategies).
 *
 *   6. When a call is made to this proxy contract (and no matching function is found in the proxy's 
 *      own interface), the call data is forwarded to the current `implementation` contract using 
 *      the `delegatecall` instruction. This means:
 *        - The code of the `implementation` contract is executed in the context of the proxy's 
 *          state (i.e., `this` and `storage` refer to the proxy).
 *        - State changes made during the `delegatecall` take effect on the proxy's storage.
 *
 *   7. The fallback and receive functions:
 *        - `fallback()` handles calls with data that do not match an existing function signature in 
 *          the proxy. It delegates these calls to the `implementation` contract.
 *        - `receive()` is a special function invoked when the contract receives ETH with no data, 
 *          ensuring the proxy can receive ETH and log it via an event.
 *
 *   8. By design, this contract can be used as the “single point of entry” for users and other 
 *      contracts interacting with a desired implementation. As upgrades occur over the proxy’s 
 *      lifetime, the external address remains consistent, maintaining the same balance, storage, 
 *      and contract address while the logic behind it can evolve.
 */
contract Proxy {
    // ---------------------------------------------------------------------------------------------
    // Special storage slot constants
    // ---------------------------------------------------------------------------------------------
    
    /**
     * @dev Implementation address stored in a special storage slot to avoid collisions.
     */
    bytes32 private constant IMPLEMENTATION_SLOT = 0x652fd83420df23d709030982db43cb11d1556d813f6109abdd59e62f58f82b1d;
    
    /**
     * @dev Proxy owner address stored in a special storage slot to avoid collisions.
     */
    bytes32 private constant PROXY_OWNER_SLOT = 0xb5c81169e7c43eaa0eef12f9f9484b95bb29e4436138cc181b687890c3bcb384;
    
    /**
     * @dev Previous implementation address stored in a special storage slot to avoid collisions.
     */
    bytes32 private constant PREVIOUS_IMPLEMENTATION_SLOT = 0xad70118ac2aa95a9982941609489a4485c04e1e1fa7b24eecb80ce8e9c8a94d1;

    // ---------------------------------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------------------------------

    /**
     * @dev Emitted whenever the proxy ownership is transferred.
     */
    event ProxyOwnershipTransferred(address indexed previousProxyOwner, address indexed newProxyOwner);

    /**
     * @dev Emitted whenever the implementation is updated.
     */
    event ImplementationChanged(address indexed previousImplementation, address indexed newImplementation); 

    /**
     * @dev Emitted whenever the contract receives ETH.
     */
    event ReceivedETH(address indexed sender, uint256 amount, uint256 timestamp);

    // ---------------------------------------------------------------------------------------------
    // Immutable and internal variables
    // ---------------------------------------------------------------------------------------------

    /**
     * @dev The timestamp when the contract was deployed (immutable).
     */
    uint256 private immutable _DEPLOYED_AT;

    /**
     * @dev The contract's name, provided in the constructor.
     */
    string private _contractName;

    // ---------------------------------------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------------------------------------

    /**
     * @dev Contract constructor.
     * @param contractName_ The name of the contract which will be hashed and stored.
     * @param implementation_ The initial implementation address that will be used for delegatecalls.
     */
    constructor(string memory contractName_, address implementation_) {
        _setProxyOwner(msg.sender);
        _setImplementation(implementation_);
        _DEPLOYED_AT = block.timestamp;
        _contractName = contractName_;
    }

    // ---------------------------------------------------------------------------------------------
    // Internal functions to set special slots
    // ---------------------------------------------------------------------------------------------

    /**
     * @dev Internal function to store a new proxy owner in the PROXY_OWNER_SLOT.
     * @param newProxyOwner The address of the new proxy owner.
     */
    function _setProxyOwner(address newProxyOwner) private {
        bytes32 slot = PROXY_OWNER_SLOT;
        assembly {
            sstore(slot, newProxyOwner)
        }
    }

    /**
     * @dev Internal function to store a new implementation in the IMPLEMENTATION_SLOT.
     *      This function also preserves the old implementation in the PREVIOUS_IMPLEMENTATION_SLOT.
     * @param newImplementation The address of the new implementation contract.
     */
    function _setImplementation(address newImplementation) private {
        bytes32 slot = IMPLEMENTATION_SLOT;
        bytes32 prevSlot = PREVIOUS_IMPLEMENTATION_SLOT;
        assembly {
            let currentImpl := sload(slot)
            sstore(prevSlot, currentImpl)
            sstore(slot, newImplementation)
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Public view functions
    // ---------------------------------------------------------------------------------------------

    /**
     * @notice Returns the address of the current proxy owner.
     * @return o The proxy owner's address stored in PROXY_OWNER_SLOT.
     */
    function proxyOwner() public view returns (address o) {
        bytes32 slot = PROXY_OWNER_SLOT;
        assembly {
            o := sload(slot)
        }
    }

    /**
     * @notice Returns the previous implementation address stored before the last update.
     * @return impl The previous implementation address.
     */
    function getPreviousImplementation() public view returns (address impl) {
        bytes32 slot = PREVIOUS_IMPLEMENTATION_SLOT;
        assembly {
            impl := sload(slot)
        }
    }

    /**
     * @notice Returns the address of the current implementation.
     * @return impl The current implementation address stored in IMPLEMENTATION_SLOT.
     */
    function getCurrentImplementation() public view returns (address impl) {
        bytes32 slot = IMPLEMENTATION_SLOT;
        assembly {
            impl := sload(slot)
        }
    }

    /**
     * @notice Returns the timestamp recorded at the moment of contract deployment.
     * @return The block timestamp when this contract was deployed.
     */
    function getDeployTimestamp() external view returns (uint256) {
        return _DEPLOYED_AT;
    }

    /**
     * @notice Returns the keccak256 hash of the contract name provided in the constructor.
     * @return The keccak256 hash of the stored contract name.
     */
    function getContractHashName() external view returns (bytes32) {
        return keccak256(abi.encodePacked(_contractName));
    }

    // ---------------------------------------------------------------------------------------------
    // External functions to update state
    // ---------------------------------------------------------------------------------------------

    /**
     * @notice Transfers the proxy ownership of the contract to a new address.
     * @param newProxyOwner The address of the new proxy owner.
     */
    function transferProxyOwnership(address newProxyOwner) external {
        require(msg.sender == proxyOwner(), "Proxy: not authorized to transfer ownership");
        address oldProxyOwner = proxyOwner();
        _setProxyOwner(newProxyOwner);
        emit ProxyOwnershipTransferred(oldProxyOwner, newProxyOwner);
    }

    /**
     * @notice Updates the implementation address used by the proxy.
     * @param newImplementation The address of the new implementation contract.
     */
    function updateImplementation(address newImplementation) external {
        require(msg.sender == proxyOwner(), "Proxy: not authorized to update implementation");
        address oldImplementation = getCurrentImplementation();
        _setImplementation(newImplementation);
        emit ImplementationChanged(oldImplementation, newImplementation);
    }

    // ---------------------------------------------------------------------------------------------
    // Fallback functions
    // ---------------------------------------------------------------------------------------------

    /**
     * @dev Fallback function that delegates all calls to the current implementation.
     *      It copies the calldata into memory and uses `delegatecall` to the stored implementation.
     */
    fallback() external payable {
        bytes32 slot = IMPLEMENTATION_SLOT;
        assembly {
            // Load free memory pointer
            let freePtr := mload(0x40)

            // Load the implementation address from the slot
            let impl := sload(slot)

            // Copy the calldata to memory starting at freePtr
            calldatacopy(freePtr, 0, calldatasize())

            // Perform delegatecall
            let result := delegatecall(gas(), impl, freePtr, calldatasize(), 0, 0)

            // Copy the returned data
            returndatacopy(freePtr, 0, returndatasize())

            // Check the result and revert or return accordingly
            switch result
            case 0 {
                revert(freePtr, returndatasize())
            }
            default {
                return(freePtr, returndatasize())
            }
        }
    }

    /**
     * @dev Receive function to accept ETH. This function will emit an event with the sender, 
     *      the value received, and the current block timestamp.
     */
    receive() external payable {
        emit ReceivedETH(msg.sender, msg.value, block.timestamp);
    }
}