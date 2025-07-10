// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity >=0.7.0 <0.9.0;

/// @title  A callback proxy of Gnosis Safe's multisignature wallet : 0.8.7-default-optimization 200
/// @author deployer - <deployer@tokenpocket.pro>
contract TokenPocketCallbackProxyV2 {

    address public executor;
    mapping(address => bool) public authorizedSafeProxyFactory;

    event SafeCreationByTokenPocket(address creator, address proxy, address singleton, bytes initializer, uint256 saltNonce);
    event DefaultSafeCreationByTokenPocket(address creator, address safeProxy);

    constructor() {
        executor = msg.sender;
    }

    fallback() external { 
        require(authorizedSafeProxyFactory[msg.sender], "Invalid safeProxyFactory called");
        emit DefaultSafeCreationByTokenPocket(tx.origin, msg.sender);
    }

    function proxyCreated(
        address proxy,
        address _singleton,
        bytes calldata initializer,
        uint256 saltNonce
    ) external onlyAuthorizedSafeProxyFactory {
        emit SafeCreationByTokenPocket(tx.origin, proxy, _singleton, initializer, saltNonce);
    }

    modifier onlyAuthorizedSafeProxyFactory() {
        require(authorizedSafeProxyFactory[msg.sender], "Invalid safeProxyFactory called");
        _;
    }

    function updateAuthorizedSafeProxyFactory(address[] calldata safeProxyFactories, bool[] calldata isAuthorized) public {
        require(msg.sender == executor, "No executer role");
        require(isAuthorized.length == safeProxyFactories.length, "invalid len");
        for (uint i; i < safeProxyFactories.length; i++) {
            authorizedSafeProxyFactory[safeProxyFactories[i]] = isAuthorized[i];
        }
    }
}