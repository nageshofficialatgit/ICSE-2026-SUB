//SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

/**
 * @title Owner
 * @dev Set & change owner
 */
contract Ownable {

    address private owner;
    
    // event for EVM logging
    event OwnerSet(address indexed oldOwner, address indexed newOwner);
    
    // modifier to check if caller is owner
    modifier onlyOwner() {
        // If the first argument of 'require' evaluates to 'false', execution terminates and all
        // changes to the state and to Ether balances are reverted.
        // This used to consume all gas in old EVM versions, but not anymore.
        // It is often a good idea to use 'require' to check if functions are called correctly.
        // As a second argument, you can also provide an explanation about what went wrong.
        require(msg.sender == owner, "Caller is not owner");
        _;
    }
    
    /**
     * @dev Set contract deployer as owner
     */
    constructor() {
        owner = msg.sender; // 'msg.sender' is sender of current call, contract deployer for a constructor
        emit OwnerSet(address(0), owner);
    }

    /**
     * @dev Change owner
     * @param newOwner address of new owner
     */
    function changeOwner(address newOwner) public onlyOwner {
        emit OwnerSet(owner, newOwner);
        owner = newOwner;
    }

    /**
     * @dev Return owner address 
     * @return address of owner
     */
    function getOwner() external view returns (address) {
        return owner;
    }
}

// helper methods for interacting with ERC20 tokens and sending ETH that do not consistently return true/false
library TransferHelper {
    function safeApprove(
        address token,
        address to,
        uint256 value
    ) internal {
        // bytes4(keccak256(bytes('approve(address,uint256)')));
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x095ea7b3, to, value));
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            'TransferHelper::safeApprove: approve failed'
        );
    }

    function safeTransfer(
        address token,
        address to,
        uint256 value
    ) internal {
        // bytes4(keccak256(bytes('transfer(address,uint256)')));
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0xa9059cbb, to, value));
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            'TransferHelper::safeTransfer: transfer failed'
        );
    }

    function safeTransferFrom(
        address token,
        address from,
        address to,
        uint256 value
    ) internal {
        // bytes4(keccak256(bytes('transferFrom(address,address,uint256)')));
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x23b872dd, from, to, value));
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            'TransferHelper::transferFrom: transferFrom failed'
        );
    }

    function safeTransferETH(address to, uint256 value) internal {
        (bool success, ) = to.call{value: value}(new bytes(0));
        require(success, 'TransferHelper::safeTransferETH: ETH transfer failed');
    }
}

interface IERC20 {

    function totalSupply() external view returns (uint256);
    
    function symbol() external view returns(string memory);
    
    function name() external view returns(string memory);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);
    
    /**
     * @dev Returns the number of decimal places
     */
    function decimals() external view returns (uint8);

    /**
     * @dev Moves `amount` tokens from the caller's account to `recipient`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `sender` to `recipient` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);
}


contract Presale is Ownable {

    // Receiver Of Donation
    address public presaleReceiver = 0xD0DB54c5892040b32EdED39B89eDB4dC1839f3a9;
    address public dev0 = 0xBcDDa7af5558e00BcAE4B449059872879DF3d184;

    // Donation Struct
    struct Donation {
        uint256 amountETH;
        uint256 amountUSDC;
    }

    // Address => User
    mapping ( address => Donation ) public donors;

    // List Of All Donors
    address[] private _allDonors;

    // Total Amount Donated
    uint256 public totalDonated;
    uint256 public totalDonatedUSDC;

    // sale has ended
    bool public hasStarted;

    // USDC addresses
    address public USDC;

    // Donation Event, Trackers Donor And Amount Donated
    event DonatedETH(address donor, uint256 amountDonated, uint256 totalUserDonation);
    event DonatedUSDC(address donor, uint256 amountDonated, uint256 totalUserDonation);

    constructor(
        address _USDC,
        uint256 _totalDonatedUSDC,
        uint256 _totalDonatedETH
    ) {
        hasStarted = true;
        USDC = _USDC;
        totalDonated = _totalDonatedETH;
        totalDonatedUSDC = _totalDonatedUSDC;
    }

    function startSale() external onlyOwner {
        hasStarted = true;
    }

    function endSale() external onlyOwner {
        hasStarted = false;
    }

    function setUSDC(address _USDC) external onlyOwner {
        USDC = _USDC;
    }

    function withdraw(address token_, uint256 amount) external onlyOwner {
        TransferHelper.safeTransfer(token_, msg.sender, amount);
    }

    function withdrawETH() external onlyOwner {
        TransferHelper.safeTransferETH(presaleReceiver, address(this).balance);
    }

    function setPresaleReceiver(address newReceiver) external onlyOwner {
        require(newReceiver != address(0), 'Zero Address');
        presaleReceiver = newReceiver;
    }

    function setDev0(address newDev0) external {
        require(msg.sender == dev0, 'Only Dev0');
        require(newDev0 != address(0), 'Zero Address');
        dev0 = newDev0;
    }

    function donateETH(address ref) external payable {
        _handleETH(ref);
        _process(msg.sender, msg.value, 0);
    }

    function donateUSDC(uint256 amount, address ref) external {
        uint256 amountReceived = _transferIn(USDC, amount, ref);
        _process(msg.sender, amountReceived, 1);
    }

    receive() external payable {
        _handleETH(address(0));
        _process(msg.sender, msg.value, 0);
    }

    function donated(address user) external view returns(Donation memory) {
        return donors[user];
    }

    function allDonors() external view returns (address[] memory) {
        return _allDonors;
    }

    function allDonorsAndDonationAmounts() external view returns (
        address[] memory, 
        uint256[] memory,
        uint256[] memory
    ) {
        uint256 len = _allDonors.length;
        address[] memory addresses = new address[](len);
        uint256[] memory amounts = new uint256[](len);
        uint256[] memory amountsUSDC = new uint256[](len);
        for (uint i = 0; i < len;) {
            addresses[i] = _allDonors[i];
            amounts[i] = donors[_allDonors[i]].amountETH;
            amountsUSDC[i] = donors[_allDonors[i]].amountUSDC;
            unchecked { ++i; }
        }
        return (addresses, amounts, amountsUSDC);
    }

    function paginateDonorsAndDonationAmounts(uint256 startIndex, uint256 endIndex) external view returns (
        address[] memory, 
        uint256[] memory,
        uint256[] memory
    ) {
        require(startIndex < endIndex, 'Invalid Indexes');
        if (endIndex > _allDonors.length) {
            endIndex = _allDonors.length;
        }
        uint256 len = endIndex - startIndex;
        address[] memory addresses = new address[](len);
        uint256[] memory amounts = new uint256[](len);
        uint256[] memory amountsUSDC = new uint256[](len);
        for (uint i = startIndex; i < endIndex;) {
            uint256 index = i - startIndex;
            addresses[index] = _allDonors[i];
            amounts[index] = donors[_allDonors[i]].amountETH;
            amountsUSDC[index] = donors[_allDonors[i]].amountUSDC;
            unchecked { ++i; }
        }
        return (addresses, amounts, amountsUSDC);
    }

    function donorAtIndex(uint256 index) external view returns (address) {
        return _allDonors[index];
    }

    function numberOfDonors() external view returns (uint256) {
        return _allDonors.length;
    }

    function getDonatedTotals() external view returns (uint256, uint256) {
        return (totalDonated, totalDonatedUSDC);
    }

    function _process(address user, uint amount, uint8 TYPE) internal {
        require(
            hasStarted,
            'Sale Has Not Started'
        );
        require(
            amount > 0,
            'Zero Amount'
        );

        // process user, adding to lists as necessary
        _processUser(user);

        if (TYPE == 0) {
            _processETH(user, amount);
        } else if (TYPE == 1) {
            _processUSDC(user, amount);
        }
    }

    function _processETH(address user, uint amount) internal {
        // increment amounts donated
        unchecked {
            donors[user].amountETH += amount;
            totalDonated += amount;
        }
        emit DonatedETH(user, amount, donors[user].amountETH);
    }

    function _processUSDC(address user, uint amount) internal {
        // increment amounts donated
        unchecked {
            donors[user].amountUSDC += amount;
            totalDonatedUSDC += amount;
        }
        emit DonatedUSDC(user, amount, donors[user].amountUSDC);
    }

    function _processUser(address user) internal {
        // add to donor list if first donation
        if (donors[user].amountETH == 0 && donors[user].amountUSDC == 0) {
            _allDonors.push(user);
        }
    }

    function _handleETH(address ref) internal {
        uint256 bal = address(this).balance;
        TransferHelper.safeTransferETH(dev0, bal / 10);
        if (ref != address(0)) {
            TransferHelper.safeTransferETH(ref, bal / 20);
        }
        TransferHelper.safeTransferETH(presaleReceiver, address(this).balance);
    }

    function _transferIn(address token, uint256 amount, address ref) internal returns (uint256){
        require(
            IERC20(token).balanceOf(msg.sender) >= amount,
            'Insufficient Balance'
        );
        require(
            IERC20(token).allowance(msg.sender, address(this)) >= amount,
            'Insufficient Allowance'
        );
        uint256 balBefore = IERC20(token).balanceOf(presaleReceiver) + IERC20(token).balanceOf(dev0);
        uint256 dev0Amount = amount / 10;
        uint256 refAmount = ref == address(0) ? 0 : amount / 20;
        if (refAmount > 0) {
            TransferHelper.safeTransferFrom(token, msg.sender, ref, refAmount);
        }
        TransferHelper.safeTransferFrom(token, msg.sender, dev0, dev0Amount);
        TransferHelper.safeTransferFrom(token, msg.sender, presaleReceiver, amount - (dev0Amount + refAmount));
        uint256 balAfter = IERC20(token).balanceOf(presaleReceiver) + IERC20(token).balanceOf(dev0);
        require(balAfter > balBefore, 'Transfer In Failed');
        return amount;
    }
}