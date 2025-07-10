/**
 *Submitted for verification at Etherscan.io on 2025-01-18
*/

/**
 *Submitted for verification at Etherscan.io on 2025-01-16
*/

/**
 *Submitted for verification at Etherscan.io on 2024-01-29
*/

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.2 <0.9.0;

interface IVault {
	function isAllocator(address a) external view returns(bool);
} 

contract PanicController {
	bool public pauseReallocation;

	bytes32[] public badMarketIds;
	address[] public badCollaterals;

	IVault public constant VAULT = IVault(0x0F359FD18BDa75e9c49bC027E7da59a4b01BF32a);
	

	function panicOn() public {
		require(VAULT.isAllocator(msg.sender), "! allocator");

		pauseReallocation = true;
	}

	function panicOff() public {
		require(VAULT.isAllocator(msg.sender), "! allocator");

		pauseReallocation = false;		
	}

	function setCustomPanicWithdraw(address[] memory badCol, bytes32[] memory badIds) public {
		require(VAULT.isAllocator(msg.sender), "! allocator");

		badMarketIds = badIds;
		badCollaterals = badCol;
	}

	function resetCustoumPanic() public {
		require(VAULT.isAllocator(msg.sender), "! allocator");

		delete badMarketIds;
		delete 	badCollaterals;
	}
}