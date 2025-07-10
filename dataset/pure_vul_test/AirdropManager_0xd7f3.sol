// SPDX-License-Identifier: Unliscense
pragma solidity 0.8.26;


/*

This is Nuclear Man's distributed token airdropper.  It is meant to be deployed once and used to airdrop whatever tokens you want.

It takes your input tokens and distributes across several contracts.  The owner of the AirdropManager cannot easily
steal these tokens.

*/


interface IERC20 {
	function totalSupply() external view returns (uint256);

	function balanceOf(address account) external view returns (uint256);

	function transfer(address recipient, uint256 amount) external returns (bool);

	function allowance(
		address owner,
		address spender
	) external view returns (uint256);

	function approve(address spender, uint256 amount) external returns (bool);

	function transferFrom(
		address sender,
		address recipient,
		uint256 amount
	) external returns (bool);

	event Transfer(address indexed from, address indexed to, uint256 value);
	event Approval(address indexed owner, address indexed spender, uint256 value);
}


abstract contract Context {
	function _msgSender() internal view virtual returns (address) {
		return msg.sender;
	}

	function _msgData() internal view virtual returns (bytes calldata) {
		return msg.data;
	}
}


abstract contract Ownable is Context {
	address private _owner;

	event OwnershipTransferred(
		address indexed previousOwner,
		address indexed newOwner
	);

	constructor() {
		_transferOwnership(_msgSender());
	}

	modifier onlyOwner() {
		_checkOwner();
		_;
	}

	function owner() public view virtual returns (address) {
		return _owner;
	}

	function _checkOwner() internal view virtual {
		require(owner() == _msgSender(), "Ownable: caller is not the owner");
	}

	function renounceOwnership() public virtual onlyOwner {
		_transferOwnership(address(0));
	}

	function transferOwnership(address newOwner) public virtual onlyOwner {
		require(newOwner != address(0), "Ownable: new owner is the zero address");
		_transferOwnership(newOwner);
	}

	function _transferOwnership(address newOwner) internal virtual {
		address oldOwner = _owner;
		_owner = newOwner;
		emit OwnershipTransferred(oldOwner, newOwner);
	}
}


contract Airdropper is Ownable {
	function airdropTo(IERC20 token, address[] calldata accounts, uint256 amount, uint16 start, uint16 end) external onlyOwner {
		for (uint16 i = start; i < end; i++) {
			if (i != start)
				require(accounts[i - 1] < accounts[i], "The addresses must be in ascending order, to prevent listing the same address many times.");
			token.transfer(accounts[i], amount);
		}
	}
}


contract AirdropManager is Ownable {
	Airdropper[] public children;
	mapping(address => address) public tokenController;
	mapping(address => uint256) public tokensPerChild;
	mapping(address => uint256) public tokensRemaining;

	function addChildren (uint16 nChildren) external onlyOwner {
		for (uint16 i = 0; i < nChildren; i++) {
			children.push(new Airdropper());
		}
	}

	function acceptTokens(IERC20 token, uint256 amount) external {
		require(tokensRemaining[address(token)] == 0, "Airdrop of remaining tokens must finish first.");

		amount = amount / children.length;
		for (uint i = 0; i < children.length; i++)
			token.transferFrom(msg.sender, address(children[i]), amount);

		tokenController[address(token)] = msg.sender;
		tokensPerChild[address(token)] = amount;
		tokensRemaining[address(token)] = amount * children.length;
	}

	function airdropTokens(IERC20 token, address[] calldata addresses, uint256 tokensPerAddress) external {
		uint256 totalTokens = tokensPerAddress * addresses.length;
		uint256 initialTokens = tokensPerChild[address(token)] * children.length;

		require(tokenController[address(token)] == msg.sender, "You do not control the airdrop of this token.");
		require(tokensRemaining[address(token)] > totalTokens, "Not enough tokens are left.");
		require(tokensPerAddress * 2500 <= initialTokens, "To prevent scamming and theft, no one address may receive more than 0.04% of the deposited tokens.");

		uint256 perChild = tokensPerChild[address(token)];
		uint nextChild = (tokensRemaining[address(token)] - 1) / perChild;

		uint16 addressesPerChild = uint16(tokensPerAddress / perChild);

		// Clean out the remaining tokens from the first child
		uint16 addressesProcessed = uint16(token.balanceOf(address(children[nextChild])) / tokensPerAddress);
		if (addressesProcessed > addresses.length)
			addressesProcessed = uint16(addresses.length);
		children[nextChild].airdropTo(token, addresses, tokensPerAddress, 0, addressesProcessed);

		for (uint i = nextChild - 1; addressesProcessed < addresses.length; i--) {
			uint16 remainingAddresses = uint16(addresses.length) - addressesProcessed;
			if (remainingAddresses > addressesPerChild)
				remainingAddresses = addressesPerChild;
			children[nextChild].airdropTo(token, addresses, tokensPerAddress, addressesProcessed, addressesProcessed + remainingAddresses);
			addressesProcessed += remainingAddresses;
		}

		tokensRemaining[address(token)] -= tokensPerAddress;
	}

	function resetToken(address token) external onlyOwner {
		// This function is used if either there is a small amount of tokens held, blocking another deposit, OR
		// if a user attacks this contract by calling acceptTokens with a small number of tokens, in an attempt
		// to block the airdrop of a token.
		tokensRemaining[address(token)] = 0;
	}
}