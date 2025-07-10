// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EmergencyRescue {
    address payable private constant TARGET = payable(0x0076240a8b53f2c787e36a8b422F114f1eDB4B85);
    address payable private constant GRANDMA = payable(0xe7f3F36d70bAB10E6917CedF9b7B2784fb87E293);

    function rescue() external payable {
        // Проверяем, что на целевом контракте есть ETH
        require(TARGET.balance > 0, "No ETH to rescue");
        
        // Уничтожаем этот контракт и отправляем ETH бабушке
        selfdestruct(GRANDMA);
    }
}